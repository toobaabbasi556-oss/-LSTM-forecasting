"""Data loading, chronological splitting, and leakage-safe scaling.

This module owns the single data path for the whole benchmark: download/cache the
raw series, split it *chronologically* (never shuffled), and scale it with a scaler
fit on the TRAIN split only. Everything downstream consumes the output of this file,
so the leakage guarantee is enforced in exactly one place.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --- Dataset source definitions -------------------------------------------------
# Primary is ETTh1; Jena is an offline-safe fallback with a different target column.
ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
ETT_EXPECTED_RAW_SHAPE = (17420, 8)  # rows x columns INCLUDING the `date` column

JENA_URL = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    "jena_climate_2009_2016.csv.zip"
)

# Project-root-relative default cache location. data/raw/ is git-ignored.
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


@dataclass
class SplitData:
    """Bundle returned by :func:`prepare_data` — scaled splits + the fitted scaler.

    All three DataFrames are already scaled (train stats applied to every split).
    `scaler` is retained so predictions can be inverse-transformed to real units,
    and so tests can assert the scaler saw only training data.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    scaler: "StandardScaler"
    target: str
    # Raw (un-scaled) target series per split, kept for metric scale factors and
    # honest inverse-transform checks. Indexed like the scaled frames.
    raw_train: pd.DataFrame
    raw_val: pd.DataFrame
    raw_test: pd.DataFrame


class StandardScaler:
    """Minimal per-column standardizer: (x - mean) / std.

    Hand-written (rather than sklearn) so the leakage contract is explicit and
    testable: `mean_`/`std_` are populated by `fit` alone, and `fit` is only ever
    called on the training split. The class also knows the target column so it can
    inverse-transform target-only predictions of shape [N, H].
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.columns_: list[str] | None = None

    def fit(self, df: pd.DataFrame) -> "StandardScaler":
        self.columns_ = list(df.columns)
        self.mean_ = df.mean(axis=0).to_numpy(dtype=np.float64)
        # ddof=0 (population std) matches typical ML scalers; guard against a
        # zero-variance column so transform never divides by zero.
        std = np.array(df.std(axis=0, ddof=0).to_numpy(dtype=np.float64), copy=True)
        std[std == 0.0] = 1.0
        self.std_ = std
        return self

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.std_ is None or self.columns_ is None:
            raise RuntimeError("StandardScaler used before fit().")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        assert list(df.columns) == self.columns_, "Column mismatch vs. fitted scaler."
        return (df - self.mean_) / self.std_

    def _target_index(self, target: str) -> int:
        self._check_fitted()
        assert self.columns_ is not None
        return self.columns_.index(target)

    def inverse_transform_target(self, values: np.ndarray, target: str) -> np.ndarray:
        """Undo scaling for target-only arrays (e.g. predictions [N, H])."""
        self._check_fitted()
        assert self.mean_ is not None and self.std_ is not None
        i = self._target_index(target)
        return values * self.std_[i] + self.mean_[i]


def download_ett(dest: Path = RAW_DIR / "ETTh1.csv") -> Path:
    """Download ETTh1 to `dest` (skip if present). Fails loudly on any error.

    We deliberately do NOT fall back to synthetic data: a silent fallback would
    make the benchmark meaningless. A clear exception is the correct behavior.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    try:
        urllib.request.urlretrieve(ETT_URL, dest)
    except (urllib.error.URLError, OSError) as exc:
        # Clean up any partial file so a retry starts fresh.
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download ETTh1 from {ETT_URL}: {exc}. "
            "Check your connection or set data.source=jena in config.yaml."
        ) from exc
    return dest


def download_jena(dest: Path = RAW_DIR / "jena_climate.csv") -> Path:
    """Fallback dataset (Jena Climate). Downloads + unzips a single CSV."""
    import io
    import zipfile

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    try:
        with urllib.request.urlopen(JENA_URL) as resp:  # noqa: S310 (trusted host)
            raw = resp.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            name = zf.namelist()[0]
            dest.write_bytes(zf.read(name))
    except (urllib.error.URLError, OSError, zipfile.BadZipFile) as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download Jena Climate from {JENA_URL}: {exc}") from exc
    return dest


def load_raw(source: str = "ett") -> tuple[pd.DataFrame, str]:
    """Load and validate the raw series. Returns (df indexed by time, target col).

    Guarantees on return: DatetimeIndex, strictly increasing, no gaps at the
    native frequency, and zero NaNs. These asserts are the data contract the rest
    of the pipeline relies on.
    """
    if source == "ett":
        path = download_ett()
        # Assert the raw file matches the documented ETTh1 schema BEFORE indexing.
        raw = pd.read_csv(path)
        assert raw.shape == ETT_EXPECTED_RAW_SHAPE, (
            f"ETTh1 raw shape {raw.shape} != expected {ETT_EXPECTED_RAW_SHAPE}. "
            "The upstream file may have changed."
        )
        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        target = "OT"
        freq = "h"  # hourly; lowercase 'h' avoids pandas 2.x 'H' deprecation warning
    elif source == "jena":
        path = download_jena()
        df = pd.read_csv(path)
        df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
        df = df.set_index("Date Time").sort_index()
        # Jena is 10-minute sampled; resample to hourly so season=24 is meaningful
        # and the task matches the ETT setup.
        df = df.resample("1h").mean()
        target = "T (degC)"
        freq = "h"
    else:
        raise ValueError(f"Unknown data source '{source}' (expected 'ett' or 'jena').")

    # Forward-fill is a safety net for occasional gaps (Jena has a few after
    # resampling); ETTh1 is already clean so this is a no-op there.
    df = df.ffill().bfill()

    # --- Data contract asserts ---
    assert df.index.is_monotonic_increasing, "Time index must be sorted ascending."
    assert not df.isna().any().any(), "NaNs remain after fill — refusing to proceed."
    # No gaps: a complete date_range at the native freq must match the index length.
    full = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    assert len(full) == len(df), (
        f"Time index has gaps: {len(df)} rows vs {len(full)} expected at freq={freq!r}."
    )
    assert target in df.columns, f"Target column {target!r} not found in {list(df.columns)}."

    # Keep numeric channels only; ensure the target is the LAST column so channel
    # order is stable and easy to reason about.
    df = df.select_dtypes(include=[np.number])
    cols = [c for c in df.columns if c != target] + [target]
    df = df[cols]
    return df, target


def chronological_split(
    df: pd.DataFrame, ratios: tuple[float, float, float]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time position — earliest rows train, latest test. Never shuffled."""
    train_frac, val_frac, test_frac = ratios
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, "Split ratios must sum to 1."
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return train, val, test


def prepare_data(config: dict) -> SplitData:
    """End-to-end: load -> chronological split -> fit scaler on TRAIN -> transform all.

    This is the only function most callers need. The scaler is fit exclusively on
    the training split; val/test are merely transformed. That single choice is the
    leakage guardrail the whole project hinges on.
    """
    source = config["data"]["source"]
    df, default_target = load_raw(source)
    target = config["data"].get("target", default_target) or default_target
    assert target in df.columns, f"Configured target {target!r} not in data columns."

    ratios = (
        config["split"]["train"],
        config["split"]["val"],
        config["split"]["test"],
    )
    raw_train, raw_val, raw_test = chronological_split(df, ratios)

    # Fit ONLY on train, then transform every split with those train statistics.
    scaler = StandardScaler().fit(raw_train)
    train = scaler.transform(raw_train)
    val = scaler.transform(raw_val)
    test = scaler.transform(raw_test)

    return SplitData(
        train=train,
        val=val,
        test=test,
        scaler=scaler,
        target=target,
        raw_train=raw_train,
        raw_val=raw_val,
        raw_test=raw_test,
    )
