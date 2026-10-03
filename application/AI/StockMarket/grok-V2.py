import argparse
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib

# Force a non-interactive backend so the script can save figures on a
# headless server or inside a notebook kernel without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - import guard for a clear message
    raise SystemExit(
        "yfinance is required. Install dependencies with: "
        "pip install torch yfinance pandas numpy scikit-learn matplotlib"
    ) from exc


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("stock_transformer")


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a single stream handler. Safe to call more than once."""
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    )
    LOGGER.setLevel(level)
    LOGGER.addHandler(handler)
    LOGGER.propagate = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Every knob that affects data, model, and training lives here.

    Keeping configuration in one immutable-style object makes the rest of
    the pipeline easy to construct and easy to override from the CLI.
    """

    ticker: str = "AAPL"
    start: str = "2015-01-01"
    end: Optional[str] = None  # None means "up to today" inside yfinance
    lookback: int = 60
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    batch_size: int = 64
    epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    num_workers: int = 0  # 0 is the portable default (Windows / notebooks)
    output_dir: str = "."
    # Feature column that the model is trained to forecast one step ahead.
    target_column: str = "Close"


# ---------------------------------------------------------------------------
# Reproducibility and device selection
# ---------------------------------------------------------------------------

class RuntimeEnvironment:
    """Seeds RNGs and picks CUDA when it is actually usable."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.device = self._select_device()
        self.seed_everything()

    @staticmethod
    def _select_device() -> torch.device:
        """Prefer CUDA, but never crash when the build has no GPU.

        torch.cuda.is_available() is the correct check: a CUDA-enabled
        wheel can be installed on a CPU-only machine, and calling .cuda()
        unconditionally would raise.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Benchmark finds a good conv/attention algorithm for fixed shapes.
            torch.backends.cudnn.benchmark = True
            LOGGER.info("CUDA device: %s", torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            LOGGER.info("CUDA not available; training on CPU.")
        return device

    def seed_everything(self) -> None:
        """Seed Python, NumPy, and PyTorch (CPU and CUDA)."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        # Deterministic algorithms can be slower and are not required here.
        # We still disable benchmark-unfriendly nondeterminism where cheap.
        os.environ["PYTHONHASHSEED"] = str(self.seed)


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

class YahooFinanceDownloader:
    """Thin wrapper around yfinance.download with column normalisation.

    Recent yfinance releases sometimes return a MultiIndex even for a
    single ticker (level 0 = field, level 1 = ticker). Downstream code
    expects a flat OHLCV frame, so this class always flattens columns.
    """

    REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

    def __init__(self, ticker: str, start: str, end: Optional[str] = None) -> None:
        self.ticker = ticker.upper().strip()
        self.start = start
        self.end = end

    def download(self) -> pd.DataFrame:
        """Download daily bars and return a clean OHLCV DataFrame.

        auto_adjust=True applies split and dividend adjustments so the
        close series is continuous. progress=False keeps logs readable.
        """
        LOGGER.info(
            "Downloading %s from Yahoo Finance (%s -> %s)",
            self.ticker,
            self.start,
            self.end or "today",
        )
        frame = yf.download(
            self.ticker,
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if frame is None or frame.empty:
            raise RuntimeError(
                f"Yahoo Finance returned no rows for ticker '{self.ticker}'. "
                "Check the symbol and the date range."
            )

        frame = self._flatten_columns(frame)
        missing = [col for col in self.REQUIRED_COLUMNS if col not in frame.columns]
        if missing:
            raise RuntimeError(
                f"Downloaded frame is missing columns {missing}. "
                f"Got columns: {list(frame.columns)}"
            )

        frame = frame.loc[:, list(self.REQUIRED_COLUMNS)].copy()
        frame = frame.sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        # Volume can arrive as float; indicators below only need numeric.
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna()
        if len(frame) < 100:
            raise RuntimeError(
                f"Only {len(frame)} rows remain for {self.ticker}. "
                "Choose a longer history."
            )
        LOGGER.info(
            "Downloaded %d rows from %s to %s",
            len(frame),
            frame.index.min().date(),
            frame.index.max().date(),
        )
        return frame

    @staticmethod
    def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
        """Collapse a possible MultiIndex down to field names."""
        if isinstance(frame.columns, pd.MultiIndex):
            # Field is level 0 for the current yfinance layout
            # ('Close', 'AAPL'). If a future layout flips that, fall back
            # to whichever level contains 'Close'.
            level0 = frame.columns.get_level_values(0)
            if "Close" in set(level0):
                frame.columns = level0
            else:
                frame.columns = frame.columns.get_level_values(-1)
        frame.columns = [str(col) for col in frame.columns]
        return frame


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

class TechnicalFeatureEngineer:
    """Build the model feature matrix from raw OHLCV bars.

    Indicators are implemented with pandas so the script does not depend
    on TA-Lib (which needs a C library). Warm-up NaNs from rolling and
    EWM windows are dropped once, after every feature has been added.

    Features
    --------
    Open, High, Low, Close, Volume
    MA10            : 10-day simple moving average of Close
    EMA12           : 12-day exponential moving average of Close
    EMA26           : 26-day EMA, used by MACD and also exposed raw
    RSI14           : Wilder-style 14-day relative strength index
    MACD            : EMA12 - EMA26
    MACD_Signal     : 9-day EMA of MACD
    MACD_Hist       : MACD - signal
    BB_PctB         : Bollinger %B (20, 2)
    ROC10           : 10-day rate of change
    Volume_MA10     : 10-day average volume
    Daily_Return    : 1-day percent change of Close
    """

    def transform(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Return a feature frame aligned to the original trading calendar."""
        close = ohlcv["Close"]
        volume = ohlcv["Volume"]

        features = pd.DataFrame(index=ohlcv.index)
        features["Open"] = ohlcv["Open"]
        features["High"] = ohlcv["High"]
        features["Low"] = ohlcv["Low"]
        features["Close"] = close
        features["Volume"] = volume

        features["MA10"] = close.rolling(window=10, min_periods=10).mean()
        features["EMA12"] = close.ewm(span=12, adjust=False).mean()
        features["EMA26"] = close.ewm(span=26, adjust=False).mean()
        features["RSI14"] = self._rsi(close, period=14)

        macd_line, macd_signal, macd_hist = self._macd(close)
        features["MACD"] = macd_line
        features["MACD_Signal"] = macd_signal
        features["MACD_Hist"] = macd_hist
        features["BB_PctB"] = self._bollinger_percent_b(close, window=20, n_std=2.0)
        features["ROC10"] = close.pct_change(periods=10)
        features["Volume_MA10"] = volume.rolling(window=10, min_periods=10).mean()
        features["Daily_Return"] = close.pct_change(periods=1)

        before = len(features)
        features = features.replace([np.inf, -np.inf], np.nan).dropna()
        LOGGER.info(
            "Engineered %d features; dropped %d warm-up rows; %d rows remain",
            features.shape[1],
            before - len(features),
            len(features),
        )
        return features

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Wilder RSI.

        Average gain / loss use an exponential window with alpha = 1/period,
        which matches the common recursive Wilder smoothing used in charting
        packages. A zero average-loss is replaced so RS does not explode;
        those rows are later dropped if they are still non-finite.
        """
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def _macd(
        close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Standard MACD triple: line, signal, histogram."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - macd_signal
        return macd_line, macd_signal, macd_hist

    @staticmethod
    def _bollinger_percent_b(
        close: pd.Series, window: int = 20, n_std: float = 2.0
    ) -> pd.Series:
        """Bollinger %B: where close sits inside the band.

        0 means the close is on the lower band, 1 means it is on the upper
        band. Values outside [0, 1] are valid (a close outside the bands).
        """
        mid = close.rolling(window=window, min_periods=window).mean()
        std = close.rolling(window=window, min_periods=window).std()
        upper = mid + n_std * std
        lower = mid - n_std * std
        width = (upper - lower).replace(0.0, np.nan)
        return (close - lower) / width


# ---------------------------------------------------------------------------
# Sequence dataset
# ---------------------------------------------------------------------------

class StockSequenceDataset(Dataset):
    """Sliding windows of scaled features -> next-day scaled close.

    Sample i uses rows [i, i + lookback) as the encoder input and the
    close at index i + lookback as the target. The target day is not
    inside the window, so the model cannot see the label in the input.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, lookback: int) -> None:
        if features.ndim != 2:
            raise ValueError("features must have shape (n_rows, n_features)")
        if len(features) != len(targets):
            raise ValueError("features and targets must have the same length")
        if len(features) <= lookback:
            raise ValueError(
                f"Need more than {lookback} rows to build a sequence, got {len(features)}"
            )
        self.features = features.astype(np.float32, copy=False)
        self.targets = targets.astype(np.float32, copy=False)
        self.lookback = lookback

    def __len__(self) -> int:
        # Last valid start index is len - lookback - 1, so count is
        # len - lookback.
        return len(self.features) - self.lookback

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.features[index : index + self.lookback]
        target = self.targets[index + self.lookback]
        return (
            torch.from_numpy(window),
            torch.tensor(target, dtype=torch.float32),
        )


class TimeSeriesPreprocessor:
    """Chronological split, leakage-free scaling, and DataLoader construction.

    Scalers are fit on the training slice only. Validation and test rows
    are transformed with those frozen statistics. The split is by time,
    never shuffled, because a random split would leak future prices into
    training.
    """

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.split_bounds: Dict[str, Tuple[int, int]] = {}

    def build(
        self, features: pd.DataFrame
    ) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DatetimeIndex]:
        """Fit scalers, cut sequences, and return train/val/test loaders.

        The returned DatetimeIndex is the full feature index. Callers
        recover test dates from split_bounds after sequence alignment.
        """
        if self.config.target_column not in features.columns:
            raise KeyError(
                f"Target column '{self.config.target_column}' is missing "
                f"from features: {list(features.columns)}"
            )

        self.feature_columns = list(features.columns)
        values = features[self.feature_columns].to_numpy(dtype=np.float64)
        target = features[self.config.target_column].to_numpy(dtype=np.float64)
        n_rows = len(features)

        train_end, val_end = self._split_indices(n_rows)
        self.split_bounds = {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, n_rows),
        }
        LOGGER.info(
            "Row split train=%d val=%d test=%d",
            train_end,
            val_end - train_end,
            n_rows - val_end,
        )

        # Fit only on training rows. Transform the whole frame afterwards
        # so sequence windows that start near a split boundary stay valid.
        # Sequence datasets below are still cut so a training window never
        # contains a validation row (see _window_range).
        self.feature_scaler.fit(values[:train_end])
        self.target_scaler.fit(target[:train_end].reshape(-1, 1))

        scaled_features = self.feature_scaler.transform(values).astype(np.float32)
        scaled_target = self.target_scaler.transform(target.reshape(-1, 1)).ravel()
        scaled_target = scaled_target.astype(np.float32)

        loaders = {}
        for name, (start, end) in self.split_bounds.items():
            # Include `lookback` rows of context before `start` so the first
            # prediction of val/test is the first row of that split, while
            # the label itself still falls inside the split.
            context_start = max(0, start - self.config.lookback)
            dataset = StockSequenceDataset(
                scaled_features[context_start:end],
                scaled_target[context_start:end],
                self.config.lookback,
            )
            # Drop windows whose label index falls before `start`.
            label_offset = context_start
            valid_indices = [
                i
                for i in range(len(dataset))
                if start <= (label_offset + i + self.config.lookback) < end
            ]
            subset = torch.utils.data.Subset(dataset, valid_indices)
            shuffle = name == "train"
            loaders[name] = DataLoader(
                subset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=self.config.num_workers,
                drop_last=False,
                pin_memory=torch.cuda.is_available(),
            )
            LOGGER.info("%s sequences: %d", name, len(subset))
            if len(subset) == 0:
                raise RuntimeError(
                    f"{name} split produced no sequences. "
                    "Increase history or reduce lookback."
                )

        return loaders["train"], loaders["val"], loaders["test"], features.index

    def _split_indices(self, n_rows: int) -> Tuple[int, int]:
        """Return (train_end, val_end) row indices. Test is the remainder."""
        if not 0.0 < self.config.train_ratio < 1.0:
            raise ValueError("train_ratio must be in (0, 1)")
        if not 0.0 < self.config.val_ratio < 1.0:
            raise ValueError("val_ratio must be in (0, 1)")
        if self.config.train_ratio + self.config.val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be < 1")
        train_end = int(n_rows * self.config.train_ratio)
        val_end = int(n_rows * (self.config.train_ratio + self.config.val_ratio))
        # Keep at least lookback + 1 rows in every slice so each loader
        # can emit sequences.
        minimum = self.config.lookback + 2
        if train_end < minimum or (val_end - train_end) < minimum or (n_rows - val_end) < minimum:
            raise RuntimeError(
                "Not enough rows for the requested lookback and split. "
                f"rows={n_rows}, lookback={self.config.lookback}."
            )
        return train_end, val_end

    def inverse_target(self, scaled_values: np.ndarray) -> np.ndarray:
        """Map model outputs back to price units (dollars per share)."""
        array = np.asarray(scaled_values, dtype=np.float64).reshape(-1, 1)
        return self.target_scaler.inverse_transform(array).ravel()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding from Vaswani et al. 2017.

    Added (not concatenated) to the token embeddings so the encoder can
    tell day t from day t-1. Dropout is applied after the addition, which
    is the layout used in the original Transformer.
    """

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        # If d_model is odd the last cosine column does not exist; slice matches.
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        # Shape (1, max_len, d_model) broadcasts across the batch dimension.
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (batch, seq, d_model)
        seq_len = tokens.size(1)
        tokens = tokens + self.encoding[:, :seq_len, :]
        return self.dropout(tokens)


class StockPriceTransformer(nn.Module):
    """Transformer encoder that regresses the next scaled close.

    Pipeline
    --------
    1. Linear projection from n_features to d_model.
    2. Sinusoidal positional encoding.
    3. A stack of pre-norm TransformerEncoder layers.
    4. The last time step (the most recent day) is read out.
    5. A small MLP maps that vector to a scalar prediction.

    batch_first=True keeps tensor layout as (batch, seq, feature), which
    matches the Dataset.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        self.input_projection = nn.Linear(n_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            # norm_first disables nested tensors; set the flag explicitly so
            # PyTorch does not emit a warning on every construction.
            enable_nested_tensor=False,
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier init for the projection and readout linears."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch, lookback, n_features) -> (batch,) predictions."""
        tokens = self.input_projection(features)
        tokens = self.positional_encoding(tokens)
        encoded = self.encoder(tokens)
        last_step = encoded[:, -1, :]
        return self.readout(last_step).squeeze(-1)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

@dataclass
class EpochMetrics:
    """One row of the learning curve."""

    epoch: int
    train_loss: float
    val_mse: float


class StockTrainer:
    """MSE training loop with AdamW, gradient clipping, and val MSE.

    The optimisation objective is mean squared error on the scaled close.
    That quantity is what the loss plot shows. The MSE plot shows the same
    metric computed on the validation loader, which is not used for the
    parameter update. Tracking them separately makes overfitting visible.
    """

    def __init__(
        self,
        model: StockPriceTransformer,
        config: TrainConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )
        self.history: List[EpochMetrics] = []

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> List[EpochMetrics]:
        """Train for config.epochs and return the per-epoch history."""
        LOGGER.info(
            "Training on %s for %d epochs (params=%d)",
            self.device,
            self.config.epochs,
            sum(p.numel() for p in self.model.parameters()),
        )
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            val_mse = self.evaluate_mse(val_loader)
            self.scheduler.step(val_mse)
            self.history.append(EpochMetrics(epoch, train_loss, val_mse))
            LOGGER.info(
                "epoch %03d | train loss %.6f | val MSE %.6f | lr %.2e",
                epoch,
                train_loss,
                val_mse,
                self.optimizer.param_groups[0]["lr"],
            )
        return self.history

    def _train_one_epoch(self, loader: DataLoader) -> float:
        """One full pass over the training loader. Returns mean MSE."""
        self.model.train()
        total_loss = 0.0
        total_count = 0
        for features, target in loader:
            features = features.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            prediction = self.model(features)
            loss = self.loss_fn(prediction, target)
            loss.backward()
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            batch_size = target.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
        return total_loss / max(total_count, 1)

    @torch.no_grad()
    def evaluate_mse(self, loader: DataLoader) -> float:
        """Mean squared error in scaled-price space. No gradient."""
        self.model.eval()
        total = 0.0
        count = 0
        for features, target in loader:
            features = features.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            prediction = self.model(features)
            # Sum of squares, then divide by count, so the metric is a
            # true mean even when the last batch is smaller.
            residual = (prediction - target) ** 2
            total += float(residual.sum().item())
            count += target.shape[0]
        return total / max(count, 1)

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Return (prediction, target) arrays in scaled space, in order.

        The test loader must be shuffle=False. Predictions are concatenated
        in dataset order so they line up with the test calendar.
        """
        self.model.eval()
        predictions: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        for features, target in loader:
            features = features.to(self.device, non_blocking=True)
            prediction = self.model(features)
            predictions.append(prediction.detach().cpu().numpy())
            targets.append(target.numpy())
        return np.concatenate(predictions), np.concatenate(targets)


class RegressionReport:
    """Price-space error summary used after inverse scaling."""

    @staticmethod
    def summarize(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """MAE, RMSE, MAPE, and directional accuracy in price units."""
        actual = np.asarray(actual, dtype=np.float64).ravel()
        predicted = np.asarray(predicted, dtype=np.float64).ravel()
        error = predicted - actual
        mae = float(np.mean(np.abs(error)))
        mse = float(np.mean(error ** 2))
        rmse = float(np.sqrt(mse))
        # MAPE is undefined at a zero price; equities will not hit that,
        # but the guard keeps the metric finite if a series is scaled oddly.
        denom = np.clip(np.abs(actual), 1e-8, None)
        mape = float(np.mean(np.abs(error) / denom) * 100.0)
        # Directional accuracy: did we get the sign of the day-over-day move?
        if len(actual) > 1:
            actual_dir = np.sign(np.diff(actual))
            pred_dir = np.sign(np.diff(predicted))
            directional = float(np.mean(actual_dir == pred_dir))
        else:
            directional = float("nan")
        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape_pct": mape,
            "directional_accuracy": directional,
        }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

class PredictionPlotter:
    """Write the three requested figures. Does not call plt.show()."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_actual_vs_predicted(
        self,
        dates: pd.DatetimeIndex,
        actual: np.ndarray,
        predicted: np.ndarray,
        ticker: str,
    ) -> str:
        """Overlay test-set actual close and model close. Returns the path."""
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(dates, actual, label="Actual close", color="#1f77b4", linewidth=1.4)
        ax.plot(
            dates,
            predicted,
            label="Predicted close",
            color="#d62728",
            linewidth=1.2,
            alpha=0.9,
        )
        ax.set_title(f"{ticker} test set: actual vs Transformer prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close price")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        path = os.path.join(self.output_dir, f"{ticker}_actual_vs_predicted.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        LOGGER.info("Wrote %s", path)
        return path

    def plot_loss(self, history: List[EpochMetrics], ticker: str) -> str:
        """Training loss (scaled MSE objective) against epoch."""
        epochs = [row.epoch for row in history]
        loss = [row.train_loss for row in history]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(epochs, loss, color="#2ca02c", marker="o", linewidth=1.4, label="Train loss")
        ax.set_title(f"{ticker} training loss vs epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE, scaled close)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        path = os.path.join(self.output_dir, f"{ticker}_loss_vs_epochs.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        LOGGER.info("Wrote %s", path)
        return path

    def plot_mse(self, history: List[EpochMetrics], ticker: str) -> str:
        """Validation MSE against epoch. Separate figure from the loss plot."""
        epochs = [row.epoch for row in history]
        mse = [row.val_mse for row in history]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(epochs, mse, color="#9467bd", marker="o", linewidth=1.4, label="Validation MSE")
        ax.set_title(f"{ticker} validation MSE vs epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (scaled close)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        path = os.path.join(self.output_dir, f"{ticker}_mse_vs_epochs.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        LOGGER.info("Wrote %s", path)
        return path


# ---------------------------------------------------------------------------
# Application object: wires every collaborator together
# ---------------------------------------------------------------------------

class StockPredictionApp:
    """End-to-end orchestration. main() only builds this object and runs it."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.environment = RuntimeEnvironment(config.seed)
        self.downloader = YahooFinanceDownloader(config.ticker, config.start, config.end)
        self.feature_engineer = TechnicalFeatureEngineer()
        self.preprocessor = TimeSeriesPreprocessor(config)
        self.plotter = PredictionPlotter(config.output_dir)

    def run(self) -> Dict[str, float]:
        """Download, train, evaluate, plot. Returns price-space test metrics."""
        ohlcv = self.downloader.download()
        features = self.feature_engineer.transform(ohlcv)
        train_loader, val_loader, test_loader, index = self.preprocessor.build(features)

        model = StockPriceTransformer(
            n_features=len(self.preprocessor.feature_columns),
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_layers=self.config.num_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
        )
        trainer = StockTrainer(model, self.config, self.environment.device)
        history = trainer.fit(train_loader, val_loader)

        scaled_pred, scaled_actual = trainer.predict(test_loader)
        actual = self.preprocessor.inverse_target(scaled_actual)
        predicted = self.preprocessor.inverse_target(scaled_pred)
        metrics = RegressionReport.summarize(actual, predicted)

        test_dates = self._test_dates(index, len(actual))
        self.plotter.plot_actual_vs_predicted(
            test_dates, actual, predicted, self.config.ticker
        )
        self.plotter.plot_loss(history, self.config.ticker)
        self.plotter.plot_mse(history, self.config.ticker)

        LOGGER.info(
            "Test MAE %.4f | RMSE %.4f | MAPE %.2f%% | directional acc %.3f",
            metrics["mae"],
            metrics["rmse"],
            metrics["mape_pct"],
            metrics["directional_accuracy"],
        )
        return metrics

    def _test_dates(self, index: pd.DatetimeIndex, n_predictions: int) -> pd.DatetimeIndex:
        """Dates of the test labels.

        A label at row r is the close on that row, predicted from the
        previous `lookback` rows. Test labels occupy
        [val_end, val_end + n_predictions).
        """
        val_end = self.preprocessor.split_bounds["test"][0]
        dates = index[val_end : val_end + n_predictions]
        if len(dates) != n_predictions:
            raise RuntimeError(
                f"Date alignment failed: {len(dates)} dates for {n_predictions} predictions"
            )
        return dates


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> TrainConfig:
    """Build a TrainConfig from command-line flags. Defaults match the dataclass."""
    parser = argparse.ArgumentParser(
        description="Predict the next daily close with a PyTorch Transformer."
    )
    parser.add_argument("--ticker", default="AAPL", help="Yahoo Finance symbol")
    parser.add_argument("--start", default="2015-01-01", help="History start date")
    parser.add_argument("--end", default=None, help="History end date (default: today)")
    parser.add_argument("--lookback", type=int, default=60, help="Input window in trading days")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer model width")
    parser.add_argument("--nhead", type=int, default=4, help="Attention heads")
    parser.add_argument("--layers", type=int, default=2, help="Encoder layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for the three PNG figures",
    )
    args = parser.parse_args(argv)
    return TrainConfig(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        lookback=args.lookback,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dropout=args.dropout,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point. Exits with status 1 on a handled data/training failure."""
    configure_logging()
    config = parse_args(argv)
    try:
        StockPredictionApp(config).run()
    except Exception as exc:  # noqa: BLE001 - top-level guard with a clean message
        LOGGER.error("Run failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
