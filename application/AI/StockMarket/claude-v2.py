#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 STOCK PRICE PREDICTION WITH A TRANSFORMER (PyTorch + CUDA + Yahoo Finance)
==============================================================================

WHAT THIS SCRIPT DOES
---------------------
1. Downloads the daily price history of a stock from Yahoo Finance (yfinance).
2. Adds technical-analysis features (MA10, MA20, EMA12, EMA26, RSI14, MACD,
   MACD signal line, MACD histogram, Bollinger Bands, daily return).
3. Scales the data (using ONLY the training period to avoid data leakage) and
   cuts it into sliding windows: "the last N days  ->  the next day's close".
4. Trains a Transformer-encoder regression model on the GPU (CUDA) if one is
   available (falls back to the CPU otherwise).
5. Evaluates the model on data it has never seen (the most recent ~15 %).
6. Opens THREE separate matplotlib windows:
        Window 1: actual price vs. predicted price
        Window 2: loss  vs. epochs  (training and validation)
        Window 3: MSE   vs. epochs  (training and validation)

INSTALL REQUIREMENTS
--------------------
    pip install yfinance pandas numpy matplotlib
    pip install torch --index-url https://download.pytorch.org/whl/cu121
        (pick the right CUDA build for your system at https://pytorch.org)

RUN
---
    python stock_transformer_predictor.py
    python stock_transformer_predictor.py --ticker MSFT --epochs 80
    python stock_transformer_predictor.py --help

CLASS OVERVIEW (everything is object oriented)
----------------------------------------------
    Config                 - dataclass holding every tunable setting
    Reproducibility        - fixes random seeds so runs are repeatable
    DeviceManager          - picks CUDA / CPU and prints GPU information
    StockDataDownloader    - downloads and cleans the Yahoo Finance data
    FeatureEngineer        - computes the technical indicators
    MinMaxScaler           - tiny, dependency-free min/max scaler
    TimeSeriesDataset      - torch Dataset wrapping (window, target) pairs
    StockDataModule        - scaling, windowing, splitting, DataLoaders
    PositionalEncoding     - sinusoidal position information for the model
    TransformerPredictor   - the neural network (nn.Module)
    TrainingHistory        - stores per-epoch loss / MSE values
    Trainer                - training loop, validation, early stopping
    MetricsCalculator      - MSE, RMSE, MAE, MAPE, R2, directional accuracy
    ResultsVisualizer      - draws the three matplotlib windows
    StockPredictionApp     - orchestrates the whole pipeline

DISCLAIMER
----------
This is an educational machine-learning example, NOT financial advice.
Stock prices are extremely noisy; a model that looks good on a chart can still
lose money in real trading. Never risk money based on this script alone.
==============================================================================
"""

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
    start_date: str = "2012-01-01"        # first day of history to download (YYYY-MM-DD)
    end_date: Optional[str] = None        # last day; None means "up to today"
    target_column: str = "Close"          # the price we want to predict

    # ---- windowing / splitting ---------------------------------------------
    sequence_length: int = 60             # how many past trading days the model sees
    train_ratio: float = 0.70             # first 70 % of the timeline  -> training
    val_ratio: float = 0.15               # next  15 %                  -> validation
    # (the remaining 15 % automatically becomes the untouched TEST set)

    # ---- Transformer architecture ------------------------------------------
    d_model: int = 64                     # internal embedding width (must be even)
    n_heads: int = 4                      # attention heads (d_model must divide by this)
    n_layers: int = 3                     # number of stacked encoder layers
    dim_feedforward: int = 256            # width of the feed-forward block in each layer
    dropout: float = 0.10                 # regularisation: randomly zero 10 % of activations

    # ---- training settings -------------------------------------------------
    batch_size: int = 64                  # samples per gradient step
    epochs: int = 100                      # maximum number of passes over the training data
    learning_rate: float = 5e-4           # AdamW step size
    weight_decay: float = 1e-4            # L2-style regularisation of the weights
    grad_clip: float = 1.0                # clip gradient norm to avoid exploding gradients
    huber_delta: float = 0.05             # Huber loss switches from squared to linear error here
    patience: int = 12                    # stop early if validation loss does not improve this long
    seed: int = 42                        # random seed for reproducibility


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

    def __init__(self, model: nn.Module, device: torch.device, config: Config) -> None:
        self.cfg = config
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
# 11. VISUALISATION
# ============================================================================
class ResultsVisualizer:
    """Builds the three matplotlib figures. Each opens in its OWN window."""

    @staticmethod
    def _set_window_title(fig: plt.Figure, title: str) -> None:
        """Set the OS window title; silently ignored on backends that cannot do this."""
        try:
            fig.canvas.manager.set_window_title(title)
        except Exception:
            pass

    def plot_predictions(
        self,
        dates: pd.DatetimeIndex,
        actual: np.ndarray,
        predicted: np.ndarray,
        train_end_idx: int,
        val_end_idx: int,
        metrics: Dict[str, float],
        ticker: str,
    ) -> None:
        """
        Window 1 - actual vs. predicted price.

        Top panel   : complete history; shaded zones mark train / validation / test.
        Bottom panel: zoom on the unseen TEST period, where the model is judged.
        """
        fig, (ax_full, ax_test) = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
        self._set_window_title(fig, f"{ticker} - Actual vs Predicted")

        # ---------------- top: whole timeline ----------------
        ax_full.plot(dates, actual, label="Actual price", color="#1f77b4", linewidth=1.3)
        ax_full.plot(dates, predicted, label="Predicted price", color="#ff7f0e", linewidth=1.0, alpha=0.9)
        ax_full.axvspan(dates[0], dates[train_end_idx], color="green", alpha=0.06, label="Training period")
        ax_full.axvspan(dates[train_end_idx], dates[val_end_idx], color="gold", alpha=0.12, label="Validation period")
        ax_full.axvspan(dates[val_end_idx], dates[-1], color="red", alpha=0.07, label="Test period (unseen)")
        ax_full.set_title(f"{ticker}: actual vs. predicted closing price (full history)")
        ax_full.set_ylabel("Price (USD)")
        ax_full.grid(True, alpha=0.3)
        ax_full.legend(loc="upper left")

        # ---------------- bottom: test period only ----------------
        test_dates = dates[val_end_idx:]
        test_actual = actual[val_end_idx:]
        test_pred = predicted[val_end_idx:]
        ax_test.plot(test_dates, test_actual, label="Actual price", color="#1f77b4", linewidth=1.6)
        ax_test.plot(test_dates, test_pred, label="Predicted price", color="#ff7f0e", linewidth=1.4)
        ax_test.fill_between(test_dates, test_actual, test_pred, color="gray", alpha=0.15, label="Prediction error")
        # NOTE: matplotlib treats a pair of "$" characters as math-mode delimiters,
        # so every literal dollar sign in a plot text must be escaped as "\$".
        ax_test.set_title(
            "Unseen test period  |  "
            f"RMSE \\${metrics['RMSE']:.2f}   MAE \\${metrics['MAE']:.2f}   "
            f"MAPE {metrics['MAPE_%']:.2f}%   R\u00b2 {metrics['R2']:.3f}"
        )
        ax_test.set_xlabel("Date")
        ax_test.set_ylabel("Price (USD)")
        ax_test.grid(True, alpha=0.3)
        ax_test.legend(loc="upper left")

    def plot_loss(self, history: TrainingHistory) -> None:
        """Window 2 - training and validation LOSS versus epochs."""
        epochs = np.arange(1, len(history.train_loss) + 1)
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        self._set_window_title(fig, "Loss vs Epochs")

        ax.plot(epochs, history.train_loss, label="Training loss", color="#2ca02c", linewidth=1.8)
        ax.plot(epochs, history.val_loss, label="Validation loss", color="#d62728", linewidth=1.8)
        ax.axvline(history.best_epoch, color="gray", linestyle="--", label=f"Best epoch ({history.best_epoch})")
        ax.set_yscale("log")     # log scale makes both early and late progress visible
        ax.set_title("Huber loss vs. epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (scaled prices, log scale)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    def plot_mse(self, history: TrainingHistory) -> None:
        """Window 3 - training and validation MSE versus epochs."""
        epochs = np.arange(1, len(history.train_mse) + 1)
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        self._set_window_title(fig, "MSE vs Epochs")

        ax.plot(epochs, history.train_mse, label="Training MSE", color="#1f77b4", linewidth=1.8)
        ax.plot(epochs, history.val_mse, label="Validation MSE", color="#ff7f0e", linewidth=1.8)
        ax.axvline(history.best_epoch, color="gray", linestyle="--", label=f"Best epoch ({history.best_epoch})")
        ax.set_yscale("log")
        ax.set_title("Mean squared error (MSE) vs. epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (scaled prices, log scale)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    @staticmethod
    def show() -> None:
        """Display every figure created so far. Blocks until all windows are closed."""
        plt.show()


# ============================================================================
# 12. APPLICATION (ORCHESTRATOR)
# ============================================================================
class StockPredictionApp:
    """Wires all the components together and runs the full pipeline."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.visualizer = ResultsVisualizer()

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

        trainer = Trainer(model, device, cfg)
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

        # --- 9) plots - three separate windows ------------------------------------
        self.visualizer.plot_predictions(
            data.target_dates, actual, predicted,
            data.train_end_idx, data.val_end_idx, test_metrics, cfg.ticker,
        )
        self.visualizer.plot_loss(history)
        self.visualizer.plot_mse(history)
        self.visualizer.show()


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