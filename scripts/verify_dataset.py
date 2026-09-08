import pandas as pd
from pathlib import Path
raw=Path("data/processed/btc_usdt_perp_1m_raw.csv.gz")
feature=Path("data/processed/btc_usdt_perp_1m_features.csv.gz")
r=pd.read_csv(raw)
f=pd.read_csv(feature)
t=pd.to_datetime(r["timestamp"], utc=True)
invalid=((r["high"] < r[["open","close","low"]].max(axis=1)) | (r["low"] > r[["open","close","high"]].min(axis=1))).sum()
gaps=((t.sort_values().diff().dropna().dt.total_seconds()/60)-1).clip(lower=0).sum()
print({"raw_bytes":raw.stat().st_size,"feature_bytes":feature.stat().st_size,"raw_rows":len(r),"feature_rows":len(f),"duplicate_timestamps":int(t.duplicated().sum()),"missing_minutes":int(gaps),"invalid_ohlc":int(invalid),"feature_columns":len(f.columns)})