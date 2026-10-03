# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
# "from __future__ import annotations" lets us use modern type-hint syntax
# (e.g. list[str], X | None) even on slightly older Python 3 versions.
from __future__ import annotations

import argparse                       # command-line argument parsing
import math                           # math.log / math.sqrt for positional encoding
import random                         # Python's built-in random generator (seeding)
import sys                            # sys.exit for clean error exits
from dataclasses import dataclass, field          # lightweight config / history classes
from typing import Dict, List, Optional, Tuple    # type hints for readability

import numpy as np                    # numerical arrays
import pandas as pd                   # tabular time-series data
import torch                          # PyTorch core
import torch.nn as nn                 # neural-network building blocks
from torch.utils.data import DataLoader, Dataset   # batching utilities

import matplotlib.pyplot as plt       # plotting windows
import yfinance as yf                 # Yahoo Finance downloader


# ============================================================================
# 1. CONFIGURATION
# ============================================================================
@dataclass
class Config:
    """
    Central place for ALL settings.

    A @dataclass automatically generates __init__, __repr__ etc. from the
    annotated fields below. Change a default here (or override it from the
    command line) and the whole pipeline follows.
    """

    # ---- data settings -----------------------------------------------------
    ticker: str = "TXN"                  # Yahoo Finance symbol, e.g. "AAPL", "MSFT", "TXN"
    start_date: str = "2020-01-01"        # first day of history to download (YYYY-MM-DD)
    end_date: Optional[str] = None        # last day; None means "up to today"
    target_column: str = "Close"          # the price we want to predict

    # ---- windowing / splitting ---------------------------------------------
    sequence_length: int = 100             # how many past trading days the model sees
    train_ratio: float = 0.70             # first 70 % of the timeline  -> training
    val_ratio: float = 0.15               # next  15 %                  -> validation
    # (the remaining 15 % automatically becomes the untouched TEST set)

    # ---- Transformer architecture ------------------------------------------
    d_model: int = 64                     # internal embedding width (must be even)
    n_heads: int = 4                      # attention heads (d_model must divide by this)
    n_layers: int = 5                     # number of stacked encoder layers
    dim_feedforward: int = 256            # width of the feed-forward block in each layer
    dropout: float = 0.0075                 # regularisation: randomly zero 10 % of activations

    # ---- training settings -------------------------------------------------
    batch_size: int = 64                  # samples per gradient step
    epochs: int = 26                      # maximum number of passes over the training data
    learning_rate: float = 2.75e-5           # AdamW step size
    weight_decay: float = 1e-4            # L2-style regularisation of the weights
    grad_clip: float = 1.0                # clip gradient norm to avoid exploding gradients
    huber_delta: float = 0.075             # Huber loss switches from squared to linear error here
    patience: int = 30                    # stop early if validation loss does not improve this long
    seed: int = 42                        # random seed for reproducibility

    # ---- live dashboard ------------------------------------------------------
    live_plot: bool = True                # update the charts while training (False = draw once at the end)
    live_update_every: int = 1            # refresh the PRICE charts every N epochs (curves refresh every epoch)


# ============================================================================
# 2. REPRODUCIBILITY HELPER
# ============================================================================
class Reproducibility:
    """Utility class that makes random behaviour repeatable between runs."""

    @staticmethod
    def set_seed(seed: int) -> None:
        """
        Seed every random-number generator we use.

        Parameters
        ----------
        seed : int
            Any integer. The same seed gives (nearly) the same results.
            GPU kernels can still introduce tiny non-deterministic differences.
        """
        random.seed(seed)                  # Python's `random` module
        np.random.seed(seed)               # NumPy
        torch.manual_seed(seed)            # PyTorch CPU
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)   # PyTorch on every GPU


# ============================================================================
# 3. DEVICE MANAGEMENT (CUDA / CPU)
# ============================================================================
class DeviceManager:
    """Chooses the compute device and prints useful hardware information."""

    @staticmethod
    def get_device() -> torch.device:
        """
        Return a CUDA device when a compatible GPU + driver is present,
        otherwise fall back to the CPU (with a visible warning).
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # cudnn.benchmark lets cuDNN auto-tune convolution/matmul algorithms
            # for the fixed input sizes we use. It is safe and often faster.
            torch.backends.cudnn.benchmark = True
            props = torch.cuda.get_device_properties(0)
            print(f"[Device] Using GPU : {torch.cuda.get_device_name(0)}")
            print(f"[Device] GPU memory: {props.total_memory / 1024 ** 3:.1f} GB")
            print(f"[Device] CUDA build: {torch.version.cuda}")
        else:
            device = torch.device("cpu")
            print("[Device] WARNING: CUDA is not available - training on the CPU.")
            print("         Install a CUDA-enabled PyTorch build to use your GPU:")
            print("         https://pytorch.org/get-started/locally/")
        return device


# ============================================================================
# 4. DATA DOWNLOAD
# ============================================================================
class StockDataDownloader:
    """Downloads daily OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance."""

    REQUIRED_COLUMNS: List[str] = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, ticker: str, start: str, end: Optional[str] = None) -> None:
        self.ticker = ticker
        self.start = start
        self.end = end

    def download(self) -> pd.DataFrame:
        """
        Fetch the data and return a clean DataFrame indexed by date.

        Raises
        ------
        RuntimeError
            If nothing could be downloaded (wrong ticker, no internet, ...).
        """
        print(f"[Data] Downloading {self.ticker} from Yahoo Finance "
              f"({self.start} -> {self.end or 'today'}) ...")

        # auto_adjust=True gives prices already adjusted for splits/dividends,
        # which is what you want for modelling. progress=False hides the bar.
        df = yf.download(
            self.ticker,
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False,
        )

        if df is None or df.empty:
            raise RuntimeError(
                f"No data returned for ticker '{self.ticker}'. "
                "Check the symbol and your internet connection."
            )

        # Recent yfinance versions return MULTI-LEVEL columns such as
        # ('Close', 'AAPL') even for a single ticker. Keep only the first level
        # ('Close') so that df["Close"] works as expected.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns.name = None

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise RuntimeError(f"Downloaded data is missing columns: {missing}")

        df = df[self.REQUIRED_COLUMNS].copy()

        # Make sure the index is a timezone-naive DatetimeIndex (simpler plotting).
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.dropna().sort_index()
        print(f"[Data] Downloaded {len(df)} trading days "
              f"({df.index[0].date()} -> {df.index[-1].date()}).")
        return df


# ============================================================================
# 5. FEATURE ENGINEERING (TECHNICAL INDICATORS)
# ============================================================================
class FeatureEngineer:
    """Adds technical-analysis indicators that give the model more context than raw prices."""

    # The exact list (and ORDER) of columns that are fed to the neural network.
    FEATURE_COLUMNS: List[str] = [
        "Open", "High", "Low", "Close", "Volume",     # raw market data
        "MA10", "MA20",                               # simple moving averages
        "EMA12", "EMA26",                             # exponential moving averages
        "MACD", "MACD_signal", "MACD_hist",           # trend / momentum
        "RSI14",                                      # relative strength index
        "BB_upper", "BB_lower",                       # Bollinger Bands
        "Return",                                     # daily percentage change
    ]

    # The EMAs need some history before they are "settled"; we discard the first
    # few rows so the network never sees those warm-up values.
    WARMUP_ROWS: int = 30

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all indicators and return a NEW DataFrame (the input is untouched).

        Indicator definitions
        ---------------------
        MA10 / MA20 : average closing price over the last 10 / 20 days.
        EMA12/EMA26 : exponential moving averages (recent days weigh more).
        MACD        : EMA12 - EMA26. Positive = short-term trend above long-term.
        MACD_signal : 9-day EMA of the MACD line (classic trigger line).
        MACD_hist   : MACD - MACD_signal (momentum of the momentum).
        RSI14       : 0..100 oscillator; >70 "overbought", <30 "oversold".
        BB_upper/lower : MA20 +/- 2 standard deviations (volatility envelope).
        Return      : close-to-close percentage change.
        """
        out = df.copy()
        close = out["Close"]

        # --- simple moving averages ------------------------------------------
        out["MA10"] = close.rolling(window=10).mean()
        out["MA20"] = close.rolling(window=20).mean()

        # --- exponential moving averages -------------------------------------
        # span=12 means alpha = 2/(12+1); adjust=False gives the recursive form
        # used by most charting software.
        out["EMA12"] = close.ewm(span=12, adjust=False).mean()
        out["EMA26"] = close.ewm(span=26, adjust=False).mean()

        # --- MACD -------------------------------------------------------------
        out["MACD"] = out["EMA12"] - out["EMA26"]
        out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
        out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

        # --- RSI --------------------------------------------------------------
        out["RSI14"] = self._compute_rsi(close, period=14)

        # --- Bollinger Bands ---------------------------------------------------
        rolling_std = close.rolling(window=20).std()
        out["BB_upper"] = out["MA20"] + 2.0 * rolling_std
        out["BB_lower"] = out["MA20"] - 2.0 * rolling_std

        # --- daily return -----------------------------------------------------
        out["Return"] = close.pct_change()

        # Rolling windows create NaNs in the first rows -> drop them, then drop
        # a few additional warm-up rows for the exponential averages.
        out = out.dropna().iloc[self.WARMUP_ROWS:]

        # Guard against any inf/NaN that could poison training.
        if not np.isfinite(out[self.FEATURE_COLUMNS].to_numpy()).all():
            raise RuntimeError("Non-finite values found in the feature matrix.")

        print(f"[Features] Using {len(self.FEATURE_COLUMNS)} features, "
              f"{len(out)} rows remain after indicator warm-up.")
        return out

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Wilder's Relative Strength Index.

        RSI = 100 - 100 / (1 + RS),  RS = average gain / average loss,
        where the averages use Wilder's smoothing (an EMA with alpha = 1/period).
        """
        delta = close.diff()                          # day-over-day price change
        gain = delta.clip(lower=0.0)                  # keep only positive moves
        loss = -delta.clip(upper=0.0)                 # keep only negative moves (as positives)

        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # If there were no losses at all, RS is infinite / undefined -> RSI is 100.
        rsi = rsi.where(avg_loss != 0.0, 100.0)
        return rsi


# ============================================================================
# 6. MIN-MAX SCALER (no scikit-learn dependency needed)
# ============================================================================
class MinMaxScaler:
    """
    Scales every column to roughly the 0..1 range:  x' = (x - min) / (max - min).

    Neural networks train far better on small, similarly-sized numbers than on
    raw values such as "Volume = 80,000,000" next to "RSI = 55".
    """

    def __init__(self) -> None:
        self.min_: Optional[np.ndarray] = None
        self.range_: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> "MinMaxScaler":
        """Learn per-column minimum and range from `data` (shape: rows x columns)."""
        self.min_ = data.min(axis=0)
        data_range = data.max(axis=0) - self.min_
        data_range[data_range == 0.0] = 1.0           # avoid division by zero for constant columns
        self.range_ = data_range
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply the learned scaling. Values outside the training range may fall outside 0..1."""
        return (data - self.min_) / self.range_

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Undo the scaling and return values in original units (e.g. US dollars)."""
        return data * self.range_ + self.min_


# ============================================================================
# 7. DATASET + DATA MODULE
# ============================================================================
class TimeSeriesDataset(Dataset):
    """
    Minimal PyTorch Dataset holding (window, target) pairs.

    features : array of shape (num_samples, sequence_length, num_features)
    targets  : array of shape (num_samples,)   - the scaled next-day close
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        # torch.from_numpy shares memory with NumPy (no copy). .float() makes sure
        # the dtype is float32, which is what GPUs are fastest with.
        self.x = torch.from_numpy(np.ascontiguousarray(features)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(targets)).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class StockDataModule:
    """
    Turns the feature DataFrame into ready-to-use DataLoaders.

    Responsibilities
    ----------------
    * chronological train / validation / test split (NO shuffling of time!)
    * fitting the scalers on TRAINING rows only (prevents look-ahead leakage)
    * building sliding windows  [t-N, ..., t-1]  ->  close[t]
    * remembering dates and real prices so results can be plotted in dollars
    """

    def __init__(self, config: Config, feature_columns: List[str]) -> None:
        self.cfg = config
        self.feature_columns = feature_columns

        # Scalers are created in prepare().
        self.feature_scaler: Optional[MinMaxScaler] = None
        self.target_scaler: Optional[MinMaxScaler] = None

        # DataLoaders, created in prepare().
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.all_loader: Optional[DataLoader] = None   # every window in time order (for plotting)

        # Bookkeeping used later for plotting and metrics.
        self.target_dates: Optional[pd.DatetimeIndex] = None   # date each prediction refers to
        self.actual_prices: Optional[np.ndarray] = None        # real close price on that date
        self.previous_close: Optional[np.ndarray] = None       # real close price the day BEFORE
        self.train_end_idx: int = 0                            # windows [0, train_end_idx) = train
        self.val_end_idx: int = 0                              # windows [train_end_idx, val_end_idx) = val
        self.num_features: int = len(feature_columns)

        # The most recent window (used to forecast the next, not-yet-happened day).
        self.latest_window: Optional[torch.Tensor] = None

    def prepare(self, df: pd.DataFrame) -> None:
        """Run the full preprocessing pipeline and build the DataLoaders."""
        cfg = self.cfg
        window = cfg.sequence_length

        features = df[self.feature_columns].to_numpy(dtype=np.float64)   # (N, F)
        close = df[cfg.target_column].to_numpy(dtype=np.float64)         # (N,)
        dates = df.index
        n_rows = len(df)

        if n_rows < window + 100:
            raise ValueError(
                f"Only {n_rows} rows available - too little for a window of "
                f"{window} days. Use an earlier --start date or a shorter --seq_len."
            )

        # ---- chronological split points (in terms of ROWS) --------------------
        train_end_row = int(n_rows * cfg.train_ratio)
        val_end_row = int(n_rows * (cfg.train_ratio + cfg.val_ratio))

        # ---- fit scalers on training rows ONLY --------------------------------
        # If we fitted on the whole history, information about the future price
        # range would leak into training and make results look better than reality.
        self.feature_scaler = MinMaxScaler().fit(features[:train_end_row])
        self.target_scaler = MinMaxScaler().fit(close[:train_end_row].reshape(-1, 1))

        features_scaled = self.feature_scaler.transform(features).astype(np.float32)
        close_scaled = self.target_scaler.transform(close.reshape(-1, 1)).ravel().astype(np.float32)

        # ---- build sliding windows --------------------------------------------
        # sliding_window_view over axis 0 gives shape (N-window+1, F, window);
        # we transpose to (N-window+1, window, F). Window i covers rows
        # i .. i+window-1 and its target is the close of row i+window.
        windows = np.lib.stride_tricks.sliding_window_view(
            features_scaled, window_shape=window, axis=0
        )
        windows = np.ascontiguousarray(windows.transpose(0, 2, 1))

        # The very last window has no "next day" target yet -> keep it aside to
        # make a genuine forecast later, and drop it from the training data.
        self.latest_window = torch.from_numpy(windows[-1:].copy()).float()
        windows = windows[:-1]                                  # (N-window, window, F)

        targets = close_scaled[window:]                         # scaled next-day close
        self.target_dates = dates[window:]                      # date of each target
        self.actual_prices = close[window:]                     # real target prices ($)
        self.previous_close = close[window - 1:-1]              # real price one day earlier ($)

        num_windows = len(targets)

        # ---- convert the ROW split points into WINDOW split points ------------
        # Window i has its target at row i+window, so a window belongs to the
        # training set when i + window < train_end_row, and so on.
        self.train_end_idx = train_end_row - window
        self.val_end_idx = val_end_row - window

        if not (0 < self.train_end_idx < self.val_end_idx < num_windows):
            raise ValueError("Split ratios leave an empty train/val/test set.")

        # ---- datasets and loaders -----------------------------------------------
        pin = torch.cuda.is_available()   # pinned host memory speeds up CPU->GPU copies

        train_ds = TimeSeriesDataset(windows[:self.train_end_idx], targets[:self.train_end_idx])
        val_ds = TimeSeriesDataset(
            windows[self.train_end_idx:self.val_end_idx], targets[self.train_end_idx:self.val_end_idx]
        )
        all_ds = TimeSeriesDataset(windows, targets)

        # Shuffling the TRAINING windows is fine: every window already contains
        # its own history, so shuffling does not leak future information.
        self.train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=pin)
        self.val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False, pin_memory=pin)
        self.all_loader = DataLoader(all_ds, batch_size=cfg.batch_size * 2, shuffle=False, pin_memory=pin)

        n_test = num_windows - self.val_end_idx
        print(f"[Data] Windows -> train: {len(train_ds)}, validation: {len(val_ds)}, test: {n_test}")
        print(f"[Data] Test period starts on {self.target_dates[self.val_end_idx].date()}")

    def inverse_target(self, scaled: np.ndarray) -> np.ndarray:
        """Convert scaled model outputs back to real prices (dollars)."""
        return self.target_scaler.inverse_transform(np.asarray(scaled).reshape(-1, 1)).ravel()


# ============================================================================
# 8. THE TRANSFORMER MODEL
# ============================================================================
class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal "position stamps" to the sequence.

    A Transformer looks at all days at once and has no built-in notion of order.
    Adding a unique sine/cosine pattern per position tells it which day came
    first and which came last.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                                     # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        # Frequencies decrease geometrically across the embedding dimensions.
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)    # even dimensions -> sine
        pe[:, 1::2] = torch.cos(position * div_term)    # odd dimensions  -> cosine

        # register_buffer: saved with the model and moved to the GPU with
        # model.to(device), but NOT treated as a trainable parameter.
        self.register_buffer("pe", pe.unsqueeze(0))     # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x has shape (batch, sequence_length, d_model)."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """
    Transformer-encoder regression network.

    Data flow
    ---------
    (batch, days, features)
        -> Linear projection to d_model          (embed the indicators)
        -> + positional encoding                 (tell it the order of days)
        -> N x TransformerEncoderLayer           (self-attention across days)
        -> take the representation of the LAST day
        -> small MLP head                        (one output: next-day scaled close)
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if d_model % 2 != 0:
            raise ValueError("d_model must be even (needed by the positional encoding).")

        # 1) Embed the raw feature vector of each day into d_model dimensions.
        self.input_projection = nn.Linear(num_features, d_model)

        # 2) Positional information.
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        # 3) Stack of Transformer encoder layers.
        #    batch_first=True -> tensors are (batch, sequence, feature).
        #    norm_first=True  -> "pre-norm" variant, trains more stably.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,   # silences a harmless warning with norm_first=True
        )

        # 4) Regression head: d_model -> d_model/2 -> 1.
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialisation of all weight matrices - a good default for Transformers."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (batch, sequence_length, num_features)

        Returns
        -------
        Tensor (batch,)  - predicted scaled close price for the day after the window.
        """
        x = self.input_projection(x)           # (B, T, d_model)
        x = self.positional_encoding(x)        # (B, T, d_model)
        x = self.encoder(x)                    # (B, T, d_model)
        last_day = x[:, -1, :]                 # (B, d_model)  summary after attending to all days
        return self.head(last_day).squeeze(-1) # (B,)


# ============================================================================
# 9. TRAINING
# ============================================================================
class TrainingCallback:
    """
    Base class for "observers" that want to be told what happens during training
    (the classic Observer / callback design pattern).

    The Trainer knows nothing about plots. It simply calls `on_epoch_end` on every
    registered callback after each epoch. A callback can then draw a chart, write a
    log file, send a notification, etc., without the Trainer being modified.

    Subclasses override only the hooks they care about; the defaults do nothing.
    """

    def on_epoch_end(self, trainer: "Trainer", epoch: int) -> None:
        """Called after every epoch. `epoch` is 1-based; `trainer.history` is up to date."""
        return None


@dataclass
class TrainingHistory:
    """Collects the numbers we plot afterwards - one entry per epoch."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_mse: List[float] = field(default_factory=list)
    val_mse: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    best_epoch: int = 0     # 1-based epoch with the lowest validation loss


class Trainer:
    """
    Handles optimisation, validation, learning-rate scheduling, early stopping
    and prediction.

    NOTE ON "LOSS" vs "MSE"
    -----------------------
    * LOSS : the Huber loss that the optimiser actually minimises. It behaves
             like MSE for small errors and like MAE for large errors, which makes
             training less sensitive to sudden price spikes.
    * MSE  : mean squared error, tracked as a separate, easy-to-interpret metric.
    Both are measured on the SCALED prices (roughly 0..1), so they are small numbers.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Config,
        callbacks: Optional[List[TrainingCallback]] = None,
    ) -> None:
        self.cfg = config
        # Observers that are notified after every epoch (e.g. the live dashboard).
        self.callbacks: List[TrainingCallback] = list(callbacks) if callbacks else []
        self.device = device
        self.model = model.to(device)    # move all weights to the GPU (if available)

        self.criterion = nn.HuberLoss(delta=config.huber_delta)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        # Halve the learning rate when the validation loss stops improving.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=4
        )
        self.history = TrainingHistory()

    # ------------------------------------------------------------------ epochs
    def _run_epoch(self, loader: DataLoader, training: bool) -> Tuple[float, float]:
        """
        Run one full pass over `loader`.

        Returns (average Huber loss, average MSE) over all samples.
        """
        self.model.train(training)     # enables dropout when training, disables it otherwise
        total_loss = 0.0
        total_sq_err = 0.0
        total_samples = 0

        # torch.no_grad() skips gradient bookkeeping during validation: faster, less memory.
        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context:
            for x_batch, y_batch in loader:
                # non_blocking=True overlaps the copy with computation (works with pinned memory).
                x_batch = x_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                predictions = self.model(x_batch)
                loss = self.criterion(predictions, y_batch)

                if training:
                    loss.backward()                                    # compute gradients
                    nn.utils.clip_grad_norm_(self.model.parameters(),  # tame huge gradients
                                             self.cfg.grad_clip)
                    self.optimizer.step()                              # update the weights

                batch_size = x_batch.size(0)
                total_loss += loss.item() * batch_size
                total_sq_err += torch.sum((predictions.detach() - y_batch) ** 2).item()
                total_samples += batch_size

        return total_loss / total_samples, total_sq_err / total_samples

    # --------------------------------------------------------------------- fit
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingHistory:
        """
        Train for up to `epochs` epochs with early stopping.

        The weights from the epoch with the best validation loss are restored
        at the end, so we keep the best model rather than the last one.
        """
        best_val_loss = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        epochs_without_improvement = 0

        print("\n[Training] Starting ...")
        print(f"{'Epoch':>5} | {'Train loss':>11} | {'Val loss':>11} | "
              f"{'Train MSE':>11} | {'Val MSE':>11} | {'LR':>9}")
        print("-" * 75)

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss, train_mse = self._run_epoch(train_loader, training=True)
            val_loss, val_mse = self._run_epoch(val_loader, training=False)

            # Let the scheduler look at the validation loss and adapt the learning rate.
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record everything for the plots.
            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.train_mse.append(train_mse)
            self.history.val_mse.append(val_mse)
            self.history.learning_rate.append(current_lr)

            marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.history.best_epoch = epoch
                epochs_without_improvement = 0
                # Keep a private copy of the best weights (clone() so later
                # training steps cannot overwrite them).
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                marker = "  <- best"
            else:
                epochs_without_improvement += 1

            print(f"{epoch:>5} | {train_loss:>11.6f} | {val_loss:>11.6f} | "
                  f"{train_mse:>11.6f} | {val_mse:>11.6f} | {current_lr:>9.2e}{marker}")

            # Tell every observer that an epoch has finished. The history (and
            # best_epoch) are already updated at this point, so a live chart
            # always sees the newest numbers.
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch)

            if epochs_without_improvement >= self.cfg.patience:
                print(f"\n[Training] Early stopping: no improvement for {self.cfg.patience} epochs.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)    # restore the best weights
        print(f"[Training] Done. Best epoch: {self.history.best_epoch} "
              f"(validation loss {best_val_loss:.6f})")
        return self.history

    # ----------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, loader: DataLoader) -> np.ndarray:
        """Return the model's (scaled) predictions for every sample in `loader`, in order."""
        self.model.eval()
        outputs: List[np.ndarray] = []
        for x_batch, _ in loader:
            x_batch = x_batch.to(self.device, non_blocking=True)
            outputs.append(self.model(x_batch).cpu().numpy())   # back to CPU for NumPy
        return np.concatenate(outputs)

    @torch.no_grad()
    def predict_window(self, window: torch.Tensor) -> float:
        """Predict (scaled) from ONE window of shape (1, sequence_length, num_features)."""
        self.model.eval()
        return float(self.model(window.to(self.device)).cpu().numpy().ravel()[0])


# ============================================================================
# 10. METRICS
# ============================================================================
class MetricsCalculator:
    """Error metrics in real price units (dollars)."""

    @staticmethod
    def compute(
        actual: np.ndarray,
        predicted: np.ndarray,
        previous_close: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Parameters
        ----------
        actual         : real prices
        predicted      : model prices
        previous_close : real price of the previous day; if given, the
                         directional accuracy is computed (did the model get the
                         up/down direction of the move right?)
        """
        error = predicted - actual
        mse = float(np.mean(error ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(error)))
        mape = float(np.mean(np.abs(error / actual)) * 100.0)       # mean abs. percentage error
        ss_res = float(np.sum(error ** 2))
        ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")  # 1.0 = perfect fit

        metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE_%": mape, "R2": r2,
                   "Directional_acc_%": float("nan")}

        if previous_close is not None:
            predicted_direction = np.sign(predicted - previous_close)
            actual_direction = np.sign(actual - previous_close)
            metrics["Directional_acc_%"] = float(np.mean(predicted_direction == actual_direction) * 100.0)
        return metrics

    @staticmethod
    def pretty_print(title: str, metrics: Dict[str, float]) -> None:
        """Print a metrics dictionary as a small aligned table."""
        print(f"\n{title}")
        print("-" * len(title))
        for name, value in metrics.items():
            if not np.isnan(value):
                print(f"  {name:<18}: {value:,.4f}")


# ============================================================================
# 11. LIVE DASHBOARD (ONE DARK WINDOW, UPDATED WHILE TRAINING)
# ============================================================================
class TrainingDashboard(TrainingCallback):
    """
    One dark-themed window with four charts that refresh after every epoch:

        +--------------------------------------------------+
        |  A) actual vs. predicted price - full history    |
        +--------------------------------------------------+
        |  B) actual vs. predicted price - unseen TEST set |
        +-------------------------+------------------------+
        |  C) loss vs. epochs     |  D) MSE vs. epochs     |
        +-------------------------+------------------------+

    HOW LIVE UPDATING WORKS
    -----------------------
    * The window is created ONCE, before training starts (matplotlib "interactive
      mode"). Static things (actual prices, shaded periods, axes, colours) are drawn
      a single time.
    * After every epoch the Trainer calls `on_epoch_end`. We then only change the
      DATA of the existing line objects (`set_data` / `set_ydata`) instead of
      redrawing everything, which is much faster, and let matplotlib repaint.
    * `plt.pause(...)` hands control to the GUI for a moment so the window repaints
      and stays responsive between epochs.

    Because the callback is just an observer, the Trainer needs no plotting code.
    """

    # ---- dark colour palette (hex colours) ------------------------------------
    BG = "#0e1117"            # figure background (near black)
    PANEL = "#161b22"         # background of each chart area
    GRID = "#30363d"          # grid lines and spines
    TEXT = "#e6edf3"          # titles, labels, tick labels
    ACTUAL = "#4fc3f7"        # actual price line (light blue)
    PREDICTED = "#ffb74d"     # predicted price line (orange)
    TRAIN = "#66bb6a"         # training curves (green)
    VALID = "#ef5350"         # validation curves (red)
    BEST = "#9aa4af"          # "best epoch" marker line (grey)

    def __init__(self, data: StockDataModule, config: Config) -> None:
        self.data = data          # gives us dates, real prices, split points and the DataLoader
        self.cfg = config

        # Matplotlib objects; created in build().
        self.fig: Optional[plt.Figure] = None
        self.ax_full: Optional[plt.Axes] = None
        self.ax_test: Optional[plt.Axes] = None
        self.ax_loss: Optional[plt.Axes] = None
        self.ax_mse: Optional[plt.Axes] = None

        # The "moving parts" - lines whose data is replaced after every epoch.
        self.line_full_pred = None
        self.line_test_pred = None
        self.error_fill = None                  # shaded area between actual and predicted
        self.loss_lines: Dict[str, object] = {}
        self.mse_lines: Dict[str, object] = {}
        self.best_lines: List[object] = []      # vertical "best epoch" markers (loss + MSE charts)
        self.status_text = None                 # headline above all charts

        self._closed = False                    # True once the user closes the window

    # ------------------------------------------------------------ style helpers
    def _style_axis(self, ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
        """Apply the dark theme (background, spines, ticks, grid, labels) to one chart."""
        ax.set_facecolor(self.PANEL)
        self._set_title(ax, title)
        ax.set_xlabel(xlabel, color=self.TEXT)
        ax.set_ylabel(ylabel, color=self.TEXT)
        ax.tick_params(colors=self.TEXT, which="both")
        for spine in ax.spines.values():
            spine.set_color(self.GRID)
        ax.grid(True, which="both", color=self.GRID, alpha=0.6, linewidth=0.6)

    def _set_title(self, ax: plt.Axes, title: str) -> None:
        """(Re)write a chart title in the dark theme's style."""
        ax.set_title(title, color=self.TEXT, fontsize=11, fontweight="bold", loc="left")

    def _style_legend(self, ax: plt.Axes, loc: str = "upper left") -> None:
        """(Re)draw a legend that matches the dark theme. Safe to call repeatedly."""
        ax.legend(loc=loc, facecolor=self.PANEL, edgecolor=self.GRID, labelcolor=self.TEXT, fontsize=9)

    @staticmethod
    def _set_window_title(fig: plt.Figure, title: str) -> None:
        """Set the OS window title; silently ignored on backends that cannot do this."""
        try:
            fig.canvas.manager.set_window_title(title)
        except Exception:
            pass

    # -------------------------------------------------------------------- build
    def build(self) -> None:
        """
        Create the window and draw everything that never changes.
        Calling it a second time does nothing (idempotent).
        """
        if self.fig is not None:
            return

        data = self.data
        dates = data.target_dates
        actual = data.actual_prices
        train_end, val_end = data.train_end_idx, data.val_end_idx
        total_epochs = self.cfg.epochs

        plt.ion()   # interactive mode: plt.show() no longer blocks, the window can be updated

        # constrained_layout keeps titles/labels from overlapping when the window is resized.
        self.fig = plt.figure(figsize=(15, 11), facecolor=self.BG, constrained_layout=True)
        self._set_window_title(self.fig, f"{self.cfg.ticker} - live Transformer training dashboard")
        self.status_text = self.fig.suptitle(
            f"{self.cfg.ticker}  |  preparing training ...",
            color=self.TEXT, fontsize=13, fontweight="bold",
        )

        # 3 rows x 2 columns grid: two wide price charts on top, loss + MSE below.
        grid = self.fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.25, 1.25, 1.0])
        self.ax_full = self.fig.add_subplot(grid[0, :])
        self.ax_test = self.fig.add_subplot(grid[1, :])
        self.ax_loss = self.fig.add_subplot(grid[2, 0])
        self.ax_mse = self.fig.add_subplot(grid[2, 1])

        nan_series = np.full(len(dates), np.nan)    # "no prediction yet" placeholder

        # ---- Chart A: full history ------------------------------------------------
        self.ax_full.plot(dates, actual, label="Actual price", color=self.ACTUAL, linewidth=1.3)
        (self.line_full_pred,) = self.ax_full.plot(
            dates, nan_series, label="Predicted price", color=self.PREDICTED, linewidth=1.2, alpha=0.95)
        self.ax_full.axvspan(dates[0], dates[train_end], color="#2e7d32", alpha=0.15, label="Training period")
        self.ax_full.axvspan(dates[train_end], dates[val_end], color="#f9a825", alpha=0.15, label="Validation period")
        self.ax_full.axvspan(dates[val_end], dates[-1], color="#c62828", alpha=0.20, label="Test period (unseen)")
        self._style_axis(self.ax_full,
                         f"{self.cfg.ticker}: actual vs. predicted closing price (full history)",
                         "", "Price (USD)")
        self._style_legend(self.ax_full)
        # Fix the y-range to the real prices so the chart does not jump around while
        # the early, still-bad predictions wander far away from reality.
        self._fix_y_range(self.ax_full, actual, padding=0.08)

        # ---- Chart B: unseen test period ------------------------------------------
        test_dates, test_actual = dates[val_end:], actual[val_end:]
        self.ax_test.plot(test_dates, test_actual, label="Actual price", color=self.ACTUAL, linewidth=1.7)
        (self.line_test_pred,) = self.ax_test.plot(
            test_dates, nan_series[val_end:], label="Predicted price", color=self.PREDICTED, linewidth=1.4)
        self._style_axis(self.ax_test, "Unseen test period  |  waiting for the first epoch ...",
                         "Date", "Price (USD)")
        self._style_legend(self.ax_test)
        self._fix_y_range(self.ax_test, test_actual, padding=0.15)

        # ---- Charts C and D: loss and MSE ------------------------------------------
        self._build_curve_axis(self.ax_loss, self.loss_lines, "Huber loss vs. epochs",
                               "Loss (scaled, log)", "Training loss", "Validation loss", total_epochs)
        self._build_curve_axis(self.ax_mse, self.mse_lines, "Mean squared error (MSE) vs. epochs",
                               "MSE (scaled, log)", "Training MSE", "Validation MSE", total_epochs)

        self._refresh()                       # show the (still empty) dashboard immediately
        try:
            plt.show(block=False)             # make sure the window is on screen
        except Exception:
            pass
        self._refresh()

    def _build_curve_axis(self, ax, store, title, ylabel, train_label, val_label, total_epochs) -> None:
        """Create the empty training/validation lines and the best-epoch marker for one curve chart."""
        # Small round markers make the very first epoch visible (a line needs 2+ points).
        (store["train"],) = ax.plot([], [], label=train_label, color=self.TRAIN, linewidth=1.8,
                                    marker="o", markersize=3.5)
        (store["val"],) = ax.plot([], [], label=val_label, color=self.VALID, linewidth=1.8,
                                  marker="o", markersize=3.5)
        best_line = ax.axvline(1, color=self.BEST, linestyle="--", linewidth=1.2, label="Best epoch")
        best_line.set_visible(False)          # hidden until the first epoch has finished
        self.best_lines.append(best_line)
        ax.set_yscale("log")                  # log scale makes both early and late progress visible
        ax.set_xlim(0.5, total_epochs + 0.5)  # fixed x-range -> the curves "grow" to the right
        # A log axis cannot be drawn while it holds no data, so give the still-empty
        # chart a sensible starting range; it is rescaled automatically after epoch 1.
        ax.set_ylim(1e-4, 1.0)
        self._style_axis(ax, title, "Epoch", ylabel)
        self._style_legend(ax, loc="upper right")

    @staticmethod
    def _fix_y_range(ax: plt.Axes, values: np.ndarray, padding: float) -> None:
        """Lock the y-axis to the range of `values` plus a relative padding."""
        low, high = float(np.min(values)), float(np.max(values))
        margin = (high - low) * padding
        ax.set_ylim(low - margin, high + margin)

    # ------------------------------------------------------------- live updates
    def on_epoch_end(self, trainer: "Trainer", epoch: int) -> None:
        """Called by the Trainer after every epoch: refresh the charts with the newest numbers."""
        if self._closed:
            return                            # the user closed the window; keep training silently
        self.build()                          # no-op if the window already exists
        history = trainer.history

        self._update_curves(history)

        # Predicting every window costs a little GPU time, so it can be thinned out
        # with `live_update_every`. Epoch 1 is always drawn.
        if epoch == 1 or epoch % max(1, self.cfg.live_update_every) == 0:
            predicted = self.data.inverse_target(trainer.predict(self.data.all_loader))
            self._update_predictions(predicted)

        self.status_text.set_text(
            f"{self.cfg.ticker}  |  training epoch {epoch}/{self.cfg.epochs}  |  "
            f"train loss {history.train_loss[-1]:.5f}   val loss {history.val_loss[-1]:.5f}   "
            f"best epoch {history.best_epoch}"
        )
        self._refresh()

    def _update_curves(self, history: TrainingHistory) -> None:
        """Replace the data of the loss/MSE lines and move the best-epoch markers."""
        epochs = np.arange(1, len(history.train_loss) + 1)
        self.loss_lines["train"].set_data(epochs, history.train_loss)
        self.loss_lines["val"].set_data(epochs, history.val_loss)
        self.mse_lines["train"].set_data(epochs, history.train_mse)
        self.mse_lines["val"].set_data(epochs, history.val_mse)

        for ax, marker in zip((self.ax_loss, self.ax_mse), self.best_lines):
            marker.set_xdata([history.best_epoch, history.best_epoch])
            marker.set_visible(True)
            marker.set_label(f"Best epoch ({history.best_epoch})")
            ax.set_autoscaley_on(True)        # set_ylim() above switched y auto-scaling off
            ax.relim()                        # recompute the data limits ...
            ax.autoscale_view(scalex=False)   # ... and rescale y only (x stays fixed)
            self._style_legend(ax, loc="upper right")

    def _update_predictions(self, predicted: np.ndarray) -> None:
        """Replace the predicted-price lines and refresh the test-period error shading and title."""
        data = self.data
        val_end = data.val_end_idx
        actual = data.actual_prices

        self.line_full_pred.set_ydata(predicted)
        self.line_test_pred.set_ydata(predicted[val_end:])

        # Re-draw the shaded "prediction error" area (fill_between cannot be edited in place).
        if self.error_fill is not None:
            self.error_fill.remove()
        self.error_fill = self.ax_test.fill_between(
            data.target_dates[val_end:], actual[val_end:], predicted[val_end:],
            color=self.BEST, alpha=0.25, label="Prediction error")

        # Test metrics shown here are for DISPLAY ONLY. They never influence training,
        # early stopping or which weights are kept (that is decided on validation data).
        metrics = MetricsCalculator.compute(actual[val_end:], predicted[val_end:])
        # NOTE: matplotlib treats a pair of "$" characters as math-mode delimiters,
        # so every literal dollar sign in a plot text must be escaped as "\\$".
        self._set_title(
            self.ax_test,
            "Unseen test period  |  "
            f"RMSE \\${metrics['RMSE']:.2f}   MAE \\${metrics['MAE']:.2f}   "
            f"MAPE {metrics['MAPE_%']:.2f}%   R\u00b2 {metrics['R2']:.3f}",
        )
        self._style_legend(self.ax_test)

    def _refresh(self) -> None:
        """Repaint the window and give the GUI a moment to process events (clicks, resizing)."""
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            # Typically raised when the window was closed while we were drawing.
            self._closed = True

    # ----------------------------------------------------------- end of training
    def finalize(self, history: TrainingHistory, predicted: np.ndarray) -> None:
        """
        Final refresh after training ended and the BEST weights were restored.
        Draws the best model's predictions (the last live frame may have shown a
        later, slightly worse epoch) and tightens the epoch axes.
        """
        self.build()                          # creates the window now if live mode was off
        if self._closed or not plt.fignum_exists(self.fig.number):
            self._closed = True
            return

        self._update_curves(history)
        n_epochs = len(history.train_loss)
        for ax in (self.ax_loss, self.ax_mse):
            ax.set_xlim(0.5, n_epochs + 0.5)  # remove the empty space if early stopping fired
        self._update_predictions(predicted)
        self.status_text.set_text(
            f"{self.cfg.ticker}  |  training finished after {n_epochs} epochs  |  "
            f"best epoch {history.best_epoch} (its weights are shown)"
        )
        self._refresh()

    def show(self) -> None:
        """Keep the window open (blocking) until the user closes it."""
        if self._closed or self.fig is None or not plt.fignum_exists(self.fig.number):
            print("[Plot] The dashboard window was closed - nothing more to show.")
            return
        plt.ioff()        # leave interactive mode so that plt.show() blocks
        plt.show()


# ============================================================================
# 12. APPLICATION (ORCHESTRATOR)
# ============================================================================
class StockPredictionApp:
    """Wires all the components together and runs the full pipeline."""

    def __init__(self, config: Config) -> None:
        self.cfg = config

    def run(self) -> None:
        cfg = self.cfg

        # --- 1) reproducibility + device --------------------------------------
        Reproducibility.set_seed(cfg.seed)
        device = DeviceManager.get_device()

        # --- 2) data download ---------------------------------------------------
        raw_df = StockDataDownloader(cfg.ticker, cfg.start_date, cfg.end_date).download()

        # --- 3) technical indicators --------------------------------------------
        engineer = FeatureEngineer()
        feature_df = engineer.add_indicators(raw_df)

        # --- 4) scaling, windowing, splitting, DataLoaders -----------------------
        data = StockDataModule(cfg, engineer.FEATURE_COLUMNS)
        data.prepare(feature_df)

        # --- 5) model + training -------------------------------------------------
        model = TransformerPredictor(
            num_features=data.num_features,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Model] Transformer with {n_params:,} trainable parameters")

        # The dashboard is an observer: the Trainer calls it after every epoch.
        # In live mode the window is opened NOW, before training starts, so you can
        # watch the curves grow and the predictions improve epoch by epoch.
        dashboard = TrainingDashboard(data, cfg)
        callbacks: List[TrainingCallback] = []
        if cfg.live_plot:
            dashboard.build()
            callbacks.append(dashboard)

        trainer = Trainer(model, device, cfg, callbacks=callbacks)
        history = trainer.fit(data.train_loader, data.val_loader)

        # --- 6) predictions for every window, converted back to dollars ----------
        predicted_scaled = trainer.predict(data.all_loader)
        predicted = data.inverse_target(predicted_scaled)
        actual = data.actual_prices
        previous = data.previous_close
        val_end = data.val_end_idx

        # --- 7) evaluation on the unseen test period ------------------------------
        test_metrics = MetricsCalculator.compute(actual[val_end:], predicted[val_end:], previous[val_end:])
        MetricsCalculator.pretty_print("TEST-SET METRICS (Transformer, unseen data, USD)", test_metrics)

        # A "naive" forecast simply says: tomorrow's price = today's price.
        # If a fancy model cannot beat this baseline, it has learned nothing useful.
        naive_metrics = MetricsCalculator.compute(actual[val_end:], previous[val_end:])
        MetricsCalculator.pretty_print("BASELINE: 'tomorrow = today' (same test period, USD)", naive_metrics)

        # --- 8) one genuine forecast for the next trading day ----------------------
        next_scaled = trainer.predict_window(data.latest_window)
        next_price = float(data.inverse_target(np.array([next_scaled]))[0])
        last_date = feature_df.index[-1].date()
        print(f"\n[Forecast] Last close on {last_date}: ${feature_df['Close'].iloc[-1]:,.2f}")
        print(f"[Forecast] Model's estimate for the NEXT trading day: ${next_price:,.2f}")
        print("[Forecast] (Educational demo only - not financial advice.)")

        # --- 9) final dashboard refresh (best weights) and keep the window open ------
        dashboard.finalize(history, predicted)
        dashboard.show()


# ============================================================================
# 13. COMMAND-LINE ENTRY POINT
# ============================================================================
def parse_arguments() -> Config:
    """Read optional command-line overrides and return a Config object."""
    defaults = Config()
    parser = argparse.ArgumentParser(
        description="Transformer-based stock price prediction (PyTorch + CUDA + yfinance)."
    )
    parser.add_argument("--ticker", default=defaults.ticker, help="Yahoo Finance symbol (default: %(default)s)")
    parser.add_argument("--start", default=defaults.start_date, help="start date YYYY-MM-DD (default: %(default)s)")
    parser.add_argument("--end", default=defaults.end_date, help="end date YYYY-MM-DD (default: today)")
    parser.add_argument("--epochs", type=int, default=defaults.epochs, help="maximum epochs (default: %(default)s)")
    parser.add_argument("--seq_len", type=int, default=defaults.sequence_length,
                        help="days of history per sample (default: %(default)s)")
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size,
                        help="batch size (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=defaults.learning_rate,
                        help="learning rate (default: %(default)s)")

    parser.add_argument("--no_live", action="store_true",
                        help="do not update the charts during training; draw them once at the end")
    parser.add_argument("--update_every", type=int, default=defaults.live_update_every,
                        help="refresh the price charts every N epochs (default: %(default)s)")

    # parse_known_args ignores unknown flags (e.g. those injected by IDEs/notebooks).
    args, _ = parser.parse_known_args()

    return Config(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        epochs=args.epochs,
        sequence_length=args.seq_len,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        live_plot=not args.no_live,
        live_update_every=args.update_every,
    )


def main() -> None:
    """Program entry point."""
    config = parse_arguments()
    try:
        StockPredictionApp(config).run()
    except (RuntimeError, ValueError) as exc:
        # Friendly error message instead of a long traceback for expected problems.
        print(f"\n[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
