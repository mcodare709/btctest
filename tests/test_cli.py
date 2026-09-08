from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from btc_perp.cli import main
from tests.test_backtest import make_market_data


class CliTests(unittest.TestCase):
    def test_model_backtest_requires_explicit_timeframe_and_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "bars.csv"
            make_market_data(80).to_csv(data, index=False)
            argv = [
                "btc_perp.cli", "backtest", "--data", str(data), "--output-dir", str(root / "out"),
                "--model", str(root / "model.cbm"),
            ]
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "requires --timeframe and --horizon-bars"):
                main()


if __name__ == "__main__":
    unittest.main()
