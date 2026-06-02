"""
Stock price prediction for TXN using a PyTorch Transformer.

Features:
  - Stationary technical + macro/sector features (SOX, peers, 10Y yield)
  - Daily or weekly log-return prediction
  - Walk-forward retraining on the test period
  - Naive baseline comparison
"""

from __future__ import annotations

import argparse
import copy
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import yfinance as yf

TICKER = "TXN"
SEQUENCE_LENGTH = 60
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
BATCH_SIZE = 64
MAX_EPOCHS = 150
WALK_FORWARD_MAX_EPOCHS = 50
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15
WALK_FORWARD_PATIENCE = 8
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 256
DROPOUT = 0.2
HUBER_DELTA = 1.0
TARGET_COLUMN = "Close"
WALK_FORWARD_RETRAIN_DAYS = 21
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Sector / macro symbols (Yahoo Finance tickers)
MACRO_TICKERS = {
    "SOX_Return": "^SOX",
    "AVGO_Return": "AVGO",
    "AMD_Return": "AMD",
    "TNX_Change": "^TNX",
}

BASE_FEATURE_COLUMNS = [
    "Return",
    "OC_Ratio",
    "HC_Ratio",
    "LC_Ratio",
    "Vol_Ratio",
    "SMA_20_Dev",
    "SMA_50_Dev",
    "EMA_12_Dev",
    "RSI_14",
    "MACD_Norm",
    "MACD_Signal_Norm",
    "BB_Width",
    "Volatility_20",
    "High_Low_Range",
]

MACRO_FEATURE_COLUMNS = list(MACRO_TICKERS.keys())


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available; using CPU.")
    return device


def flatten_yfinance(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
    return df


def download_stock_data(ticker: str, years: int = 8) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=365 * years)
    print(f"Downloading {ticker} from Yahoo Finance ({start.date()} to {end.date()})...")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    df = flatten_yfinance(df)

    if df.empty:
        raise RuntimeError(f"No data returned for ticker {ticker}.")

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    df = df[required].dropna().astype(float)
    print(f"  {ticker}: {len(df)} trading days.")
    return df


def download_macro_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Download sector index, peers, and 10Y yield; align to stock calendar."""
    start, end = index.min(), index.max()
    print("Downloading macro/sector data:")
    frames: list[pd.Series] = []

    for feat_name, symbol in MACRO_TICKERS.items():
        print(f"  {feat_name} <- {symbol}")
        raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        raw = flatten_yfinance(raw)
        if raw.empty or "Close" not in raw.columns:
            print(f"    Warning: no data for {symbol}, skipping.")
            continue

        close = raw["Close"].reindex(index).ffill()
        if feat_name == "TNX_Change":
            series = close.pct_change()
        else:
            series = close.pct_change()
        series.name = feat_name
        frames.append(series)

    if not frames:
        return pd.DataFrame(index=index)

    macro = pd.concat(frames, axis=1)
    return macro.reindex(index)


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].replace(0, np.nan)

    out["Return"] = close.pct_change()
    out["OC_Ratio"] = out["Open"] / close - 1
    out["HC_Ratio"] = out["High"] / close - 1
    out["LC_Ratio"] = out["Low"] / close - 1

    vol_sma = out["Volume"].rolling(20).mean().replace(0, np.nan)
    out["Vol_Ratio"] = out["Volume"] / vol_sma - 1

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    out["SMA_20_Dev"] = close / sma20 - 1
    out["SMA_50_Dev"] = close / sma50 - 1
    out["EMA_12_Dev"] = close / ema12 - 1

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI_14"] = (100 - (100 / (1 + rs))) / 100 - 0.5

    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["MACD_Norm"] = macd / close
    out["MACD_Signal_Norm"] = macd.ewm(span=9, adjust=False).mean() / close

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["BB_Width"] = (2 * bb_std) / bb_mid.replace(0, np.nan)
    out["Volatility_20"] = out["Return"].rolling(20).std()
    out["High_Low_Range"] = (out["High"] - out["Low"]) / close

    return out.replace([np.inf, -np.inf], np.nan)


def build_forward_log_returns(close: pd.Series, horizon: int) -> pd.Series:
    """log(close[t+horizon] / close[t]); NaN for last `horizon` rows."""
    shifted = close.shift(-horizon)
    return np.log(shifted / close.replace(0, np.nan))


def build_full_dataset(
    raw_df: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    tech = add_technical_features(raw_df)
    macro = download_macro_features(tech.index)
    combined = tech.join(macro, how="left")
    combined[MACRO_FEATURE_COLUMNS] = combined[MACRO_FEATURE_COLUMNS].ffill().bfill()

    feature_cols = BASE_FEATURE_COLUMNS + [
        c for c in MACRO_FEATURE_COLUMNS if c in combined.columns
    ]
    combined["Log_Return_Target"] = build_forward_log_returns(
        combined[TARGET_COLUMN], horizon
    )

    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
    feature_df = combined[feature_cols].copy()
    close = combined[TARGET_COLUMN].values.astype(np.float64)
    targets = combined["Log_Return_Target"].values.astype(np.float64)

    min_rows = SEQUENCE_LENGTH + horizon + 80
    if len(feature_df) < min_rows:
        raise RuntimeError(f"Not enough rows ({len(feature_df)}). Need {min_rows}+.")

    return feature_df, close, targets, feature_cols


def split_indices(n: int) -> tuple[int, int]:
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    return train_end, val_end


class StockSequenceDataset(Dataset):
    def __init__(
        self,
        scaled_features: np.ndarray,
        targets: np.ndarray,
        seq_len: int,
        start: int,
        end: int,
    ):
        self.x = scaled_features.astype(np.float32)
        self.y = targets.astype(np.float32)
        self.seq_len = seq_len
        self.start = start
        self.end = end

    def __len__(self) -> int:
        return self.end - self.start - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.start + idx
        x = self.x[i : i + self.seq_len]
        y = self.y[i + self.seq_len]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class StockTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        seq_len: int = 60,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.head(x[:, -1, :]).squeeze(-1)


@dataclass
class TrainResult:
    model: StockTransformer
    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int


@dataclass
class EvalResult:
    actuals: np.ndarray
    predictions: np.ndarray
    naive: np.ndarray
    eval_dates: pd.DatetimeIndex


def run_epoch(
    model: StockTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if train_mode:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                pred = model(x_batch)
                loss = criterion(pred, y_batch)

        total_loss += loss.item()
        batches += 1

    return total_loss / max(batches, 1)


def train_model(
    scaled_features: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    train_end: int,
    val_end: int,
    num_features: int,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    verbose: bool = True,
) -> TrainResult:
    model = StockTransformer(
        num_features=num_features,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        seq_len=SEQUENCE_LENGTH,
    ).to(device)

    train_ds = StockSequenceDataset(scaled_features, targets, SEQUENCE_LENGTH, 0, train_end)
    val_ds = StockSequenceDataset(
        scaled_features, targets, SEQUENCE_LENGTH, train_end, val_end
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        tr_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:3d}/{max_epochs}  train: {tr_loss:.6f}  "
                f"val: {va_loss:.6f}  lr: {lr:.2e}"
            )

        if stale_epochs >= patience:
            if verbose:
                print(f"  Early stop epoch {epoch} (best: {best_epoch}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(model, train_losses, val_losses, best_epoch)


def fit_scaler(feature_matrix: np.ndarray, train_end: int) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(feature_matrix[:train_end])
    return scaler


@torch.no_grad()
def predict_static_test(
    model: StockTransformer,
    scaled_features: np.ndarray,
    close_prices: np.ndarray,
    test_start: int,
    device: torch.device,
    horizon: int,
    step: int = 1,
) -> EvalResult:
    """Single model, walk test indices (step=1 daily, step=5 weekly)."""
    model.eval()
    n = len(close_prices)
    actuals: list[float] = []
    predictions: list[float] = []
    naive: list[float] = []
    eval_indices: list[int] = []

    i = test_start + SEQUENCE_LENGTH
    while i < n:
        seq = scaled_features[i - SEQUENCE_LENGTH : i]
        x = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0).to(device)
        pred_log_ret = model(x).cpu().item()

        base_idx = i - 1
        future_idx = i - 1 + horizon
        if future_idx >= n:
            break

        base_close = close_prices[base_idx]
        actual_close = close_prices[future_idx]
        pred_close = base_close * math.exp(pred_log_ret)

        predictions.append(pred_close)
        actuals.append(actual_close)
        naive.append(base_close)
        eval_indices.append(future_idx)
        i += step

    return EvalResult(
        np.array(actuals),
        np.array(predictions),
        np.array(naive),
        pd.DatetimeIndex([]),
    )


def predict_walk_forward(
    feature_matrix: np.ndarray,
    targets: np.ndarray,
    close_prices: np.ndarray,
    device: torch.device,
    test_start: int,
    horizon: int,
    step: int,
    num_features: int,
) -> tuple[EvalResult, list[int]]:
    """
    Expanding-window retrain every WALK_FORWARD_RETRAIN_DAYS during test.
    Scaler refit only at retrain points (no future leakage).
    """
    n = len(close_prices)
    actuals: list[float] = []
    predictions: list[float] = []
    naive: list[float] = []
    eval_indices: list[int] = []

    model: StockTransformer | None = None
    active_scaler: StandardScaler | None = None
    days_since_retrain = WALK_FORWARD_RETRAIN_DAYS

    i = test_start + SEQUENCE_LENGTH
    while i < n:
        future_idx = i - 1 + horizon
        if future_idx >= n:
            break

        if model is None or days_since_retrain >= WALK_FORWARD_RETRAIN_DAYS:
            history_end = i - horizon
            val_start = max(int(history_end * (1 - VAL_RATIO)), SEQUENCE_LENGTH + 50)

            active_scaler = StandardScaler()
            active_scaler.fit(feature_matrix[:history_end])
            scaled = active_scaler.transform(feature_matrix[:history_end])

            print(
                f"  Walk-forward retrain at {i} "
                f"(history 0..{history_end}, train 0..{val_start}, val {val_start}..{history_end})"
            )
            result = train_model(
                scaled,
                targets[:history_end],
                device,
                val_start,
                history_end,
                num_features,
                max_epochs=WALK_FORWARD_MAX_EPOCHS,
                patience=WALK_FORWARD_PATIENCE,
                verbose=False,
            )
            model = result.model
            days_since_retrain = 0

        assert active_scaler is not None and model is not None
        scaled_window = active_scaler.transform(feature_matrix[i - SEQUENCE_LENGTH : i])

        with torch.no_grad():
            model.eval()
            x = torch.from_numpy(scaled_window.astype(np.float32)).unsqueeze(0).to(device)
            pred_log_ret = model(x).cpu().item()

        base_close = close_prices[i - 1]
        predictions.append(base_close * math.exp(pred_log_ret))
        actuals.append(close_prices[future_idx])
        naive.append(base_close)
        eval_indices.append(future_idx)

        days_since_retrain += step
        i += step

    ev = EvalResult(np.array(actuals), np.array(predictions), np.array(naive), pd.DatetimeIndex([]))
    return ev, eval_indices


def compute_metrics(actuals: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    errors = actuals - predictions
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mape = float(np.mean(np.abs(errors / actuals)) * 100)
    direction = float(
        np.mean(np.sign(np.diff(actuals)) == np.sign(np.diff(predictions)))
        if len(actuals) > 1
        else 0.0
    )
    return {"mae": mae, "rmse": rmse, "mape": mape, "direction_acc": direction * 100}


def print_metrics_block(name: str, metrics: dict[str, float]) -> None:
    print(
        f"  {name:30s} MAE ${metrics['mae']:6.2f}  "
        f"RMSE ${metrics['rmse']:6.2f}  "
        f"MAPE {metrics['mape']:5.2f}%  "
        f"Dir.Acc {metrics['direction_acc']:5.1f}%"
    )


def plot_comparison(
    dates: pd.DatetimeIndex,
    close_prices: np.ndarray,
    test_start: int,
    results: dict[str, EvalResult],
    eval_indices_map: dict[str, list[int]],
    ticker: str,
    mode_label: str,
    train_result: TrainResult | None,
) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax_price = axes[0, 0]
    ax_price.plot(dates, close_prices, color="steelblue", linewidth=1.0, alpha=0.7, label="Close")
    ax_price.axvline(dates[test_start], color="gray", linestyle="--", alpha=0.6, label="Test start")

    colors = {"static": "crimson", "walk_forward": "darkviolet"}
    for name, ev in results.items():
        idxs = eval_indices_map[name]
        ev_dates = dates[idxs]
        ax_price.plot(
            ev_dates,
            ev.actuals,
            "o",
            markersize=2,
            alpha=0.5,
            label=f"Actual ({name})",
        )
        ax_price.plot(
            ev_dates,
            ev.predictions,
            "-",
            color=colors.get(name, "black"),
            linewidth=1.2,
            label=f"Pred ({name})",
        )

    ax_price.set_title(f"{ticker} — {mode_label} predictions")
    ax_price.set_ylabel("USD")
    ax_price.legend(loc="best", fontsize=8)
    ax_price.grid(True, alpha=0.3)

    ax_err = axes[0, 1]
    for name, ev in results.items():
        idxs = eval_indices_map[name]
        err = ev.actuals - ev.predictions
        ax_err.plot(dates[idxs], err, label=name, alpha=0.8)
    ax_err.axhline(0, color="black", linewidth=0.8)
    ax_err.set_title("Prediction error")
    ax_err.legend(fontsize=8)
    ax_err.grid(True, alpha=0.3)

    ax_metrics = axes[1, 0]
    ax_metrics.axis("off")
    lines = [f"{mode_label} — Test metrics\n"]
    for eval_name, ev in results.items():
        m = compute_metrics(ev.actuals, ev.predictions)
        n = compute_metrics(ev.actuals, ev.naive)
        lines.append(f"{eval_name}:")
        lines.append(
            f"  Model  RMSE ${m['rmse']:.2f}  MAPE {m['mape']:.2f}%  Dir {m['direction_acc']:.1f}%"
        )
        lines.append(
            f"  Naive  RMSE ${n['rmse']:.2f}  MAPE {n['mape']:.2f}%  Dir {n['direction_acc']:.1f}%"
        )
        beat = "yes" if m["rmse"] < n["rmse"] else "no"
        lines.append(f"  Beats naive RMSE: {beat}\n")
    ax_metrics.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=9, family="monospace")

    ax_loss = axes[1, 1]
    if train_result is not None:
        ax_loss.plot(train_result.train_losses, label="Train")
        ax_loss.plot(train_result.val_losses, label="Val")
        ax_loss.set_title(f"Initial training loss (best epoch {train_result.best_epoch})")
        ax_loss.legend()
    else:
        ax_loss.text(0.5, 0.5, "Walk-forward only", ha="center")
    ax_loss.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_label = mode_label.lower().replace(" ", "_")
    out_path = os.path.join(OUTPUT_DIR, f"{ticker}_{safe_label}_predictions.png")
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved: {out_path}")
    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()
    plt.close(fig)
    return out_path


def run_mode(
    raw_df: pd.DataFrame,
    device: torch.device,
    horizon: int,
    mode_name: str,
    use_walk_forward: bool,
) -> None:
    step = horizon
    feature_df, close_prices, targets, feature_cols = build_full_dataset(raw_df, horizon)
    dates = feature_df.index
    feature_matrix = feature_df.values.astype(np.float64)
    n = len(feature_matrix)
    train_end, val_end = split_indices(n)
    num_features = feature_matrix.shape[1]

    print(f"\n{'=' * 60}")
    print(f"Mode: {mode_name}  |  horizon={horizon} trading days  |  step={step}")
    print(f"Features ({len(feature_cols)}):")
    for name in feature_cols:
        print(f"  - {name}")
    print(
        f"Split — train: {train_end}, val: {val_end - train_end}, test: {n - val_end}"
    )

    scaler = fit_scaler(feature_matrix, train_end)
    scaled = scaler.transform(feature_matrix)

    print("\nInitial training (static model)...")
    train_result = train_model(
        scaled, targets, device, train_end, val_end, num_features, verbose=True
    )
    print(f"Best epoch: {train_result.best_epoch}")

    static_ev = predict_static_test(
        train_result.model, scaled, close_prices, val_end, device, horizon, step
    )

    results: dict[str, EvalResult] = {"static": static_ev}
    eval_indices_map: dict[str, list[int]] = {
        "static": [
            idx - 1 + horizon
            for idx in range(val_end + SEQUENCE_LENGTH, n, step)
            if idx - 1 + horizon < n
        ]
    }

    if use_walk_forward:
        print(f"\nWalk-forward retraining (every {WALK_FORWARD_RETRAIN_DAYS} days)...")
        wf_ev, wf_indices = predict_walk_forward(
            feature_matrix,
            targets,
            close_prices,
            device,
            val_end,
            horizon,
            step,
            num_features,
        )
        results["walk_forward"] = wf_ev
        eval_indices_map["walk_forward"] = wf_indices

    print(f"\n{mode_name} — test metrics:")
    for eval_name, ev in results.items():
        print(f"  [{eval_name}]")
        print_metrics_block("Transformer", compute_metrics(ev.actuals, ev.predictions))
        print_metrics_block("Naive", compute_metrics(ev.actuals, ev.naive))

    plot_comparison(
        dates,
        close_prices,
        val_end,
        results,
        eval_indices_map,
        TICKER,
        mode_name,
        train_result,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TXN stock transformer predictor")
    parser.add_argument(
        "--mode",
        choices=["daily", "weekly", "all"],
        default="all",
        help="Prediction horizon: 1 day, 5 trading days, or both",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        default=True,
        help="Enable walk-forward retraining on test set (default: on)",
    )
    parser.add_argument(
        "--no-walk-forward",
        action="store_true",
        help="Disable walk-forward retraining",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_walk_forward = args.walk_forward and not args.no_walk_forward
    device = get_device()

    raw_df = download_stock_data(TICKER)

    modes: list[tuple[str, int]] = []
    if args.mode in ("daily", "all"):
        modes.append(("Daily (1-day)", 1))
    if args.mode in ("weekly", "all"):
        modes.append(("Weekly (5-day)", 5))

    print(f"\nWalk-forward retraining: {'ON' if use_walk_forward else 'OFF'}")

    for mode_name, horizon in modes:
        run_mode(raw_df, device, horizon, mode_name, use_walk_forward)


if __name__ == "__main__":
    main()