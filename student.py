# student.py
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor


class Student:
    """
    Minimal baseline that predicts next-H-day cumulative log return from OHLCV.
    (1) Student responsibilities: data/feature engineering + model selection.
    (2) Tester provides the target y (log(C_{t+H}/C_t)) and walk-forward protocol.

    SCoursework essentials:
      (1) Data preprocessing (causal) from OHLCV
      (2) Feature engineering (simple technical indicators)
      (3) Model selection (small, time-aware CV for Ridge alpha)

    API expected by the tester:
      (1) fit(X_train: pd.DataFrame, y_train: pd.Series, meta: dict|None) -> self
      (2) predict(X: pd.DataFrame, meta: dict|None) -> pd.Series named 'y_pred'
    """

    def __init__(
        self,
        config=None,               # accepts None or dict of overrides
        random_state: int = 42,
        *,
        # feature knobs
        n_lags=10,                 # r_{t-1} ... r_{t-n}
        mom_windows=(10, 20, 60),  # mean of daily log-returns
        vol_window=30,             # std of daily log-returns
        sma_windows=(10, 20, 50, 200),
        ema_windows=(12, 26),
        rsi_window=14,
        # model selection knobs
        alpha_grid=(0.01, 0.1, 1.0, 10.0),
        cv_splits=3,
        min_train_points=200,
        **kwargs                   # tolerate extra kwargs from caller
    ):
        # defaults
        self.n_lags = int(n_lags)
        self.mom_windows = tuple(int(w) for w in mom_windows)
        self.vol_window = int(vol_window)
        self.sma_windows = tuple(int(w) for w in sma_windows)
        self.ema_windows = tuple(int(w) for w in ema_windows)
        self.rsi_window = int(rsi_window)

        self.alpha_grid = tuple(float(a) for a in alpha_grid)
        self.cv_splits = int(cv_splits)
        self.min_train_points = int(min_train_points)
        self.random_state = int(random_state)

        # overrides
        if isinstance(config, dict):
            for k, v in config.items():
                if hasattr(self, k):
                    setattr(self, k, v)
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # learned state
        self.pipe_ = None
        self.best_alpha_ = None
        self.fitted_ = False

    # ---------- helpers ----------

    @staticmethod
    def _close_series(X: pd.DataFrame) -> pd.Series:
        return X["Close"] if "Close" in X.columns else X.iloc[:, 0]

    @staticmethod
    def _log_returns(series: pd.Series) -> pd.Series:
        series = pd.Series(series).astype(float)
        return np.log(series / series.shift(1))

    @staticmethod
    def _rsi(close: pd.Series, window: int) -> pd.Series:
        """RSI in [0,1] using Wilder’s smoothing (causal)."""
        close = pd.Series(close).astype(float)
        diff = close.diff()
        gain = diff.clip(lower=0.0)
        loss = -diff.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 1 - (1 / (1 + rs))  # 0..1
        return rsi.fillna(0.5)

    @staticmethod
    def _finite_mean(y: pd.Series) -> float:
        """Return finite mean of y or 0.0 if none available."""
        yv = pd.Series(y).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(yv) == 0:
            return 0.0
        m = float(yv.mean())
        return m if np.isfinite(m) else 0.0

    def _make_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Leakage-safe features:
          - lagged daily log returns
          - rolling momentum (mean of log returns)
          - rolling volatility (std of log returns)
          - SMA/EMA distance (%)
          - RSI (0..1)
        """
        close = self._close_series(X).astype(float)
        lr = self._log_returns(close)

        feats = {}

        # lags r_{t-1},...,r_{t-n}
        for i in range(1, self.n_lags + 1):
            feats[f"lag{i}"] = lr.shift(i)

        # momentum (mean of daily log-returns)
        for w in self.mom_windows:
            feats[f"mom_{w}"] = lr.rolling(w, min_periods=w).mean()

        # volatility (std of daily log-returns)
        wv = self.vol_window
        feats[f"vol_{wv}"] = lr.rolling(wv, min_periods=wv).std(ddof=0)

        # SMA/EMA distances: (price - MA) / MA
        for w in self.sma_windows:
            sma = close.rolling(w, min_periods=w).mean()
            feats[f"sma_dist_{w}"] = (close - sma) / sma.replace(0, np.nan)

        for w in self.ema_windows:
            ema = close.ewm(span=w, adjust=False, min_periods=w).mean()
            feats[f"ema_dist_{w}"] = (close - ema) / ema.replace(0, np.nan)

        # RSI
        feats[f"rsi_{self.rsi_window}"] = self._rsi(close, self.rsi_window)

        F = pd.DataFrame(feats, index=X.index).replace([np.inf, -np.inf], np.nan)
        F = F.dropna()  # ensure all features present and causal
        return F

    # ---------- fit / predict ----------

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, meta=None):
        """
        Select Ridge alpha with small TimeSeriesSplit, then fit on all training data.
        Predict y_train directly (H-day cumulative log return).
        """
        F = self._make_features(X_train)

        # Fallback if we can't compute features yet (early folds)
        if F.empty:
            mean_y = self._finite_mean(y_train)
            self.pipe_ = Pipeline([("model", DummyRegressor(strategy="constant", constant=mean_y))])
            self.pipe_.fit([[0.0]], [0.0])  # fit on non-NaN dummy target
            self.best_alpha_ = None
            self.fitted_ = True
            return self

        # Align target to feature index and drop NaNs/infs
        y = y_train.reindex(F.index)
        mask = y.replace([np.inf, -np.inf], np.nan).notna()
        F, y = F.loc[mask], y.loc[mask]

        # If nothing valid after alignment, fallback to constant
        if len(y) == 0:
            mean_y = self._finite_mean(y_train)
            self.pipe_ = Pipeline([("model", DummyRegressor(strategy="constant", constant=mean_y))])
            self.pipe_.fit([[0.0]], [0.0])
            self.best_alpha_ = None
            self.fitted_ = True
            return self

        # If very short history, avoid overfitting: use constant mean of current y
        if len(F) < self.min_train_points:
            mean_y = float(y.mean()) if np.isfinite(y.mean()) else 0.0
            self.pipe_ = Pipeline([("model", DummyRegressor(strategy="constant", constant=mean_y))])
            self.pipe_.fit([[0.0]], [0.0])
            self.best_alpha_ = None
            self.fitted_ = True
            return self

        # Time-aware CV for alpha
        n_splits = min(self.cv_splits, max(2, len(F) // 200))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        best_alpha, best_mse = None, np.inf
        for a in self.alpha_grid:
            mses = []
            for tr_idx, va_idx in tscv.split(F.values):
                X_tr, X_va = F.values[tr_idx], F.values[va_idx]
                y_tr, y_va = y.values[tr_idx], y.values[va_idx]
                pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=a))])
                pipe.fit(X_tr, y_tr)
                y_hat = pipe.predict(X_va)
                mses.append(mean_squared_error(y_va, y_hat))
            avg_mse = float(np.mean(mses)) if mses else np.inf
            if avg_mse < best_mse:
                best_mse, best_alpha = avg_mse, a

        if best_alpha is None:
            best_alpha = 1.0
        self.best_alpha_ = float(best_alpha)

        # Fit final model on full training window
        self.pipe_ = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=self.best_alpha_))])
        self.pipe_.fit(F.values, y.values)
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, meta=None) -> pd.Series:
        """
        Return predictions of next-H-day cumulative log returns (regression output).
        Series index = dates where features are available; name = 'y_pred'.
        """
        F = self._make_features(X)
        if not self.fitted_ or self.pipe_ is None:
            idx = F.index if len(F) else X.index
            return pd.Series(0.0, index=idx, name="y_pred")
        if F.empty:
            return pd.Series(0.0, index=X.index, name="y_pred")
        y_hat = self.pipe_.predict(F.values)
        return pd.Series(y_hat, index=F.index, name="y_pred")
