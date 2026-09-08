"""Feature availability statistics that preserve missing-value semantics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _max_missing_period(index: pd.Index, missing: pd.Series) -> str | None:
    if not missing.any():
        return None
    groups = missing.ne(missing.shift(fill_value=False)).cumsum()
    rows = int(missing.groupby(groups).sum().max())
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        step = index.to_series().diff().dropna().median()
        if pd.notna(step):
            return str(rows * step)
    return f"{rows} rows"


def feature_availability_report(
    frame: pd.DataFrame, feature_names: Iterable[str] | None = None
) -> pd.DataFrame:
    """Report coverage while keeping NaN and numeric zero distinct."""

    names = list(feature_names) if feature_names is not None else list(frame.columns)
    rows: list[dict[str, object]] = []
    for name in names:
        values = (
            pd.to_numeric(frame[name], errors="coerce")
            if name in frame
            else pd.Series(np.nan, index=frame.index, dtype=float)
        ).replace([np.inf, -np.inf], np.nan)
        missing = values.isna()
        observed = values.loc[~missing]
        rows.append(
            {
                "feature_name": name,
                "coverage_start": observed.index[0].isoformat() if len(observed) and hasattr(observed.index[0], "isoformat") else (observed.index[0] if len(observed) else None),
                "coverage_end": observed.index[-1].isoformat() if len(observed) and hasattr(observed.index[-1], "isoformat") else (observed.index[-1] if len(observed) else None),
                "missing_rate": float(missing.mean()) if len(values) else 0.0,
                "zero_rate": float((values == 0).mean()) if len(values) else 0.0,
                "unique_count": int(observed.nunique(dropna=True)),
                "variance": float(observed.var(ddof=0)) if len(observed) else None,
                "constant": bool(len(observed) > 0 and observed.nunique(dropna=True) <= 1),
                "continuous_missing_periods": _max_missing_period(frame.index, missing),
            }
        )
    return pd.DataFrame(rows)
