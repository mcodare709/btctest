"""Write the full NaN/zero feature availability report for a CSV dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_perp.availability import feature_availability_report
from btc_perp.feature_schema import FEATURE_SCHEMAS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--schema", choices=sorted(FEATURE_SCHEMAS), default=None)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp")
    names = FEATURE_SCHEMAS[args.schema] if args.schema else None
    report = feature_availability_report(frame, names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(orient="records"), indent=2, default=str), encoding="utf-8")
    print(json.dumps({"rows": len(frame), "features": len(report), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
