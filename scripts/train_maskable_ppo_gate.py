"""Train a frozen CatBoost expected-return gate with a MaskablePPO controller."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_perp.feature_schema import CORE_FEATURES
from btc_perp.gym_env import GymRLPortfolioEnv
from btc_perp.protocol import EXECUTION_PROTOCOL_VERSION, net_horizon_returns
from btc_perp.return_policy import non_overlapping_actions
from btc_perp.rl_env import PortfolioAction, RLPortfolioConfig

EXPECTED_COLUMNS = ("long_expected_net_return", "short_expected_net_return")


def _episode_starts(rows: int, episode_steps: int, count: int) -> list[int]:
    max_start = rows - episode_steps - 1
    if max_start < 0:
        raise ValueError("validation data is shorter than one episode")
    if count <= 0:
        raise ValueError("evaluation episode count must be positive")
    return sorted(set(np.linspace(0, max_start, num=count, dtype=int).tolist()))


def _gate_counts(frame: pd.DataFrame, threshold_bps: float) -> dict[str, int]:
    threshold = threshold_bps / 10_000.0
    long_expected = frame[EXPECTED_COLUMNS[0]].to_numpy(dtype=float)
    short_expected = frame[EXPECTED_COLUMNS[1]].to_numpy(dtype=float)
    long_gate = (long_expected > threshold) & (long_expected > short_expected)
    short_gate = (short_expected > threshold) & (short_expected > long_expected)
    return {
        "long": int(long_gate.sum()),
        "short": int(short_gate.sum()),
        "flat": int((~long_gate & ~short_gate).sum()),
    }


def _full_gate_policy_metrics(
    frame: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    threshold_bps: float,
    horizon_bars: int,
) -> dict[str, object]:
    expected = frame[list(EXPECTED_COLUMNS)].to_numpy(dtype=float)
    actions = non_overlapping_actions(
        expected,
        threshold=threshold_bps / 10_000.0,
        horizon_bars=horizon_bars,
    )
    realized = actual[["long_net_return", "short_net_return"]].to_numpy(dtype=float)
    valid = np.isfinite(realized).all(axis=1)
    actions[~valid] = 0
    chosen = np.where(actions == 1, realized[:, 0], np.where(actions == -1, realized[:, 1], np.nan))
    pnl = chosen[actions != 0]
    gains = float(pnl[pnl > 0].sum()) if pnl.size else 0.0
    losses = float(-pnl[pnl < 0].sum()) if pnl.size else 0.0
    return {
        "trade_count": int(pnl.size),
        "long_count": int((actions == 1).sum()),
        "short_count": int((actions == -1).sum()),
        "win_rate": float((pnl > 0).mean()) if pnl.size else None,
        "mean_net_return": float(pnl.mean()) if pnl.size else None,
        "total_uncompounded_net_return": float(pnl.sum()),
        "profit_factor": gains / losses if losses > 0 else None,
    }


def _summarize(episodes: list[dict[str, object]]) -> dict[str, object]:
    returns = np.asarray([float(item["return"]) for item in episodes])
    drawdowns = np.asarray([float(item["drawdown"]) for item in episodes])
    return {
        "episode_count": len(episodes),
        "mean_return": float(returns.mean()),
        "median_return": float(np.median(returns)),
        "positive_episode_rate": float((returns > 0).mean()),
        "worst_return": float(returns.min()),
        "best_return": float(returns.max()),
        "mean_max_drawdown": float(drawdowns.mean()),
        "worst_max_drawdown": float(drawdowns.max()),
        "liquidation_count": int(sum(bool(item["liquidated"]) for item in episodes)),
        "trade_count": int(sum(int(item["trade_count"]) for item in episodes)),
        "entry_attempts_blocked": int(sum(int(item["entry_attempts_blocked"]) for item in episodes)),
        "episodes": episodes,
    }


def _config(args: argparse.Namespace) -> RLPortfolioConfig:
    return RLPortfolioConfig(
        episode_leverage_choices=(3.0,),
        max_entries_per_window=args.max_entries_per_window,
        entry_rate_window_minutes=args.entry_rate_window_minutes,
        expected_return_gate_enabled=True,
        min_expected_edge_bps=args.min_expected_edge_bps,
    )


def _evaluate(
    model: object,
    normalizer: object,
    frame: pd.DataFrame,
    *,
    args: argparse.Namespace,
    feature_names: tuple[str, ...],
    policy: str,
) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    episodes: list[dict[str, object]] = []
    starts = _episode_starts(len(frame), args.episode_steps, args.evaluation_episodes)
    for episode_index, start in enumerate(starts):
        environment = GymRLPortfolioEnv(
            frame,
            feature_names=feature_names,
            config=_config(args),
            episode_steps=args.episode_steps,
            random_start=False,
            fixed_start_index=start,
            fixed_max_leverage=3.0,
            seed=args.seed + episode_index,
        )
        observation, info = environment.reset(seed=args.seed + episode_index)
        terminated = truncated = False
        actions: Counter[str] = Counter()
        while not (terminated or truncated):
            mask = environment.action_masks()
            if policy == "maskable_ppo":
                normalized = normalizer.normalize_obs(observation.copy())
                action_value, _ = model.predict(normalized, action_masks=mask, deterministic=True)
                action = int(np.asarray(action_value).reshape(-1)[0])
            elif policy == "random_valid":
                action = int(rng.choice(np.flatnonzero(mask)))
            elif policy == "catboost_1x":
                if mask[PortfolioAction.LONG_1X]:
                    action = int(PortfolioAction.LONG_1X)
                elif mask[PortfolioAction.SHORT_1X]:
                    action = int(PortfolioAction.SHORT_1X)
                else:
                    action = int(PortfolioAction.HOLD)
            elif policy == "hold":
                action = int(PortfolioAction.HOLD)
            else:
                raise ValueError(f"unknown evaluation policy: {policy}")
            actions[PortfolioAction(action).name] += 1
            observation, _, terminated, truncated, info = environment.step(action)
        episodes.append(
            {
                "episode": episode_index,
                "start": frame.index[start].isoformat(),
                "end": frame.index[environment.core.cursor].isoformat(),
                "return": float(info["equity"]) / environment.core.config.initial_equity - 1.0,
                "drawdown": float(info["drawdown"]),
                "liquidated": bool(info["liquidated"]),
                "trade_count": len(environment.core.trades),
                "entry_attempts_blocked": int(info["entry_attempts_blocked"]),
                "actions": dict(actions),
            }
        )
    return _summarize(episodes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/core_1m_features.csv.gz"))
    parser.add_argument("--output-prefix", type=Path, default=Path("outputs/models/maskable_ppo_catboost_gate"))
    parser.add_argument("--tail-rows", type=int, default=600_000)
    parser.add_argument("--expert-train-rows", type=int, default=600_000)
    parser.add_argument("--validation-rows", type=int, default=120_000)
    parser.add_argument("--horizon-bars", type=int, default=10)
    parser.add_argument("--episode-steps", type=int, default=2_048)
    parser.add_argument("--evaluation-episodes", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--expert-iterations", type=int, default=51)
    parser.add_argument("--thread-count", type=int, default=10)
    parser.add_argument("--min-expected-edge-bps", type=float, default=5.0)
    parser.add_argument("--max-entries-per-window", type=int, default=5)
    parser.add_argument("--entry-rate-window-minutes", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from catboost import CatBoostRegressor
    import gymnasium
    import sb3_contrib
    import stable_baselines3
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    purge_rows = args.horizon_bars + 1
    needed = args.expert_train_rows + args.tail_rows + 2 * purge_rows
    columns = ["timestamp", "open", "high", "low", "close", *CORE_FEATURES]
    frame = pd.read_csv(args.input, compression="gzip", usecols=columns).tail(needed).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp").sort_index()
    if len(frame) < needed:
        raise ValueError("input does not contain enough expert and RL rows")
    rl_start = len(frame) - args.tail_rows
    expert_end = rl_start - purge_rows
    expert_start = expert_end - args.expert_train_rows
    if expert_start < 0:
        raise ValueError("expert_train_rows and tail_rows do not fit input")

    targets = net_horizon_returns(
        frame,
        args.horizon_bars,
        fee_rate=0.0004,
        slippage_bps=1.0,
        default_spread_bps=2.0,
    )[["long_net_return", "short_net_return"]]
    x_expert = frame.iloc[expert_start:expert_end][list(CORE_FEATURES)].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    y_expert = targets.iloc[expert_start:expert_end]
    if not y_expert.notna().all(axis=1).all():
        raise ValueError("expert target contains missing rows")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame = frame.iloc[rl_start:].copy()
    x_prediction = prediction_frame[list(CORE_FEATURES)].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    expert_artifacts: dict[str, str] = {}
    for side, target_name, prediction_name in zip(
        ("long", "short"),
        ("long_net_return", "short_net_return"),
        EXPECTED_COLUMNS,
    ):
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=args.expert_iterations,
            depth=6,
            learning_rate=0.05,
            random_seed=args.seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=args.thread_count,
        )
        model.fit(x_expert, y_expert[target_name], verbose=False)
        prediction_frame[prediction_name] = np.asarray(model.predict(x_prediction), dtype=np.float32)
        artifact = args.output_prefix.with_name(f"{args.output_prefix.name}_gate_{side}.cbm")
        model.save_model(str(artifact))
        expert_artifacts[side] = str(artifact)

    validation_start = len(prediction_frame) - args.validation_rows
    train_end = validation_start - purge_rows
    if train_end <= args.episode_steps or validation_start <= train_end:
        raise ValueError("RL split does not leave train, purge, and validation rows")
    train_frame = prediction_frame.iloc[:train_end].copy()
    validation_frame = prediction_frame.iloc[validation_start:].copy()
    features = tuple(CORE_FEATURES) + EXPECTED_COLUMNS

    def make_train_environment() -> GymRLPortfolioEnv:
        return GymRLPortfolioEnv(
            train_frame,
            feature_names=features,
            config=_config(args),
            episode_steps=args.episode_steps,
            random_start=True,
            fixed_max_leverage=3.0,
            seed=args.seed,
        )

    vector_environment = VecNormalize(
        DummyVecEnv([make_train_environment]),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    model = MaskablePPO(
        "MlpPolicy",
        vector_environment,
        learning_rate=3e-4,
        n_steps=1_024,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        policy_kwargs={"net_arch": [128, 128]},
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    model_path = args.output_prefix.with_suffix(".zip")
    normalizer_path = args.output_prefix.with_name(f"{args.output_prefix.name}_vecnormalize.pkl")
    model.save(str(args.output_prefix))
    vector_environment.save(str(normalizer_path))
    vector_environment.training = False
    vector_environment.norm_reward = False

    reports = {
        policy: _evaluate(
            model,
            vector_environment,
            validation_frame,
            args=args,
            feature_names=features,
            policy=policy,
        )
        for policy in ("maskable_ppo", "catboost_1x", "random_valid", "hold")
    }
    fixed_cost_bps = 2.0 * (0.0004 * 10_000.0 + 1.0) + 2.0
    report = {
        "experiment": "maskable-ppo-frozen-catboost-net-return-gate-v1",
        "execution_protocol": EXECUTION_PROTOCOL_VERSION,
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "sb3_contrib": sb3_contrib.__version__,
            "device": str(model.device),
        },
        "artifacts": {
            "maskable_ppo": str(model_path),
            "normalizer": str(normalizer_path),
            "catboost": expert_artifacts,
        },
        "data": {
            "source": str(args.input),
            "source_timeframe": "1min",
            "expert_train_rows": len(x_expert),
            "expert_training_start": x_expert.index[0].isoformat(),
            "expert_training_end": x_expert.index[-1].isoformat(),
            "expert_to_rl_purge_rows": purge_rows,
            "rl_train_rows": len(train_frame),
            "rl_training_start": train_frame.index[0].isoformat(),
            "rl_training_end": train_frame.index[-1].isoformat(),
            "rl_to_validation_purge_rows": purge_rows,
            "validation_rows": len(validation_frame),
            "validation_start": validation_frame.index[0].isoformat(),
            "validation_end": validation_frame.index[-1].isoformat(),
        },
        "gate": {
            "target": "separate_long_short_net_return",
            "feature_names": list(CORE_FEATURES),
            "horizon_bars": args.horizon_bars,
            "fixed_round_trip_cost_bps": fixed_cost_bps,
            "safety_margin_bps": args.min_expected_edge_bps,
            "required_gross_edge_bps": fixed_cost_bps + args.min_expected_edge_bps,
            "expert_iterations": args.expert_iterations,
            "train_distribution": _gate_counts(train_frame, args.min_expected_edge_bps),
            "validation_distribution": _gate_counts(validation_frame, args.min_expected_edge_bps),
            "full_validation_non_overlapping_policy": _full_gate_policy_metrics(
                validation_frame,
                targets.loc[validation_frame.index],
                threshold_bps=args.min_expected_edge_bps,
                horizon_bars=args.horizon_bars,
            ),
        },
        "risk": {
            "episode_max_leverage": 3.0,
            "max_entries_per_window": args.max_entries_per_window,
            "entry_rate_window_minutes": args.entry_rate_window_minutes,
            "max_drawdown": 0.20,
            "max_holding_bars": 10,
        },
        "training": {
            "algorithm": "MaskablePPO",
            "requested_total_timesteps": args.total_timesteps,
            "actual_total_timesteps": int(model.num_timesteps),
            "seed": args.seed,
            "episode_steps": args.episode_steps,
            "ent_coef": 0.0,
        },
        "validation": reports,
        "limitations": [
            "Single-seed fixed holdout baseline; not yet multi-fold walk-forward RL.",
            "Core history uses fixed 12 bps round-trip cost and lacks event-complete funding and Mark Price OHLC.",
            "CatBoost experts are frozen before all RL rows, but regime drift can make the gate sparse or stale.",
        ],
    }
    report_path = args.output_prefix.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
