"""Train a small PPO risk-controller baseline and evaluate fixed OOS episodes."""

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
from btc_perp.rl_env import PortfolioAction, RLPortfolioConfig


def _episode_starts(rows: int, episode_steps: int, count: int) -> list[int]:
    max_start = rows - episode_steps - 1
    if max_start < 0:
        raise ValueError("validation data is shorter than one episode")
    if count <= 0:
        raise ValueError("evaluation episode count must be positive")
    return sorted(set(np.linspace(0, max_start, num=count, dtype=int).tolist()))


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
        "successful_entry_count": int(sum(int(item["successful_entry_count"]) for item in episodes)),
        "entry_attempts_blocked": int(sum(int(item["entry_attempts_blocked"]) for item in episodes)),
        "episodes": episodes,
    }


def _evaluate(
    model: object,
    normalizer: object,
    frame: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
    episode_steps: int,
    episode_count: int,
    seed: int,
    policy: str,
    max_entries_per_window: int,
    entry_rate_window_minutes: float,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    episodes: list[dict[str, object]] = []
    for episode_index, start in enumerate(_episode_starts(len(frame), episode_steps, episode_count)):
        environment = GymRLPortfolioEnv(
            frame,
            feature_names=feature_names,
            config=RLPortfolioConfig(
                episode_leverage_choices=(3.0,),
                max_entries_per_window=max_entries_per_window,
                entry_rate_window_minutes=entry_rate_window_minutes,
            ),
            episode_steps=episode_steps,
            random_start=False,
            fixed_start_index=start,
            fixed_max_leverage=3.0,
            seed=seed + episode_index,
        )
        observation, info = environment.reset(seed=seed + episode_index)
        terminated = False
        truncated = False
        actions: Counter[str] = Counter()
        while not (terminated or truncated):
            if policy == "random":
                action = int(rng.integers(len(PortfolioAction)))
            elif policy == "hold":
                action = int(PortfolioAction.HOLD)
            elif policy == "ppo":
                normalized = normalizer.normalize_obs(observation.copy())
                action_value, _ = model.predict(normalized, deterministic=True)
                action = int(np.asarray(action_value).reshape(-1)[0])
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
                "successful_entry_count": int(info["successful_entry_count"]),
                "entry_attempts_blocked": int(info["entry_attempts_blocked"]),
                "actions": dict(actions),
            }
        )
    return _summarize(episodes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/core_1m_features.csv.gz"))
    parser.add_argument("--output-prefix", type=Path, default=Path("outputs/models/ppo_core_baseline"))
    parser.add_argument("--tail-rows", type=int, default=600_000)
    parser.add_argument("--validation-rows", type=int, default=120_000)
    parser.add_argument("--purge-bars", type=int, default=11)
    parser.add_argument("--episode-steps", type=int, default=2_048)
    parser.add_argument("--evaluation-episodes", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-entries-per-window", type=int, default=5)
    parser.add_argument("--entry-rate-window-minutes", type=float, default=5.0)
    args = parser.parse_args()

    import gymnasium
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    columns = ["timestamp", "open", "high", "low", "close", *CORE_FEATURES]
    frame = pd.read_csv(args.input, compression="gzip", usecols=columns)
    if args.tail_rows <= args.validation_rows + args.purge_bars + args.episode_steps:
        raise ValueError("tail_rows must leave train, purge, validation, and one full episode")
    frame = frame.tail(args.tail_rows).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp").sort_index()
    validation_start = len(frame) - args.validation_rows
    train_end = validation_start - args.purge_bars
    train_frame = frame.iloc[:train_end].copy()
    validation_frame = frame.iloc[validation_start:].copy()
    features = tuple(CORE_FEATURES)

    def make_train_environment() -> GymRLPortfolioEnv:
        return GymRLPortfolioEnv(
            train_frame,
            feature_names=features,
            config=RLPortfolioConfig(
                episode_leverage_choices=(3.0,),
                max_entries_per_window=args.max_entries_per_window,
                entry_rate_window_minutes=args.entry_rate_window_minutes,
            ),
            episode_steps=args.episode_steps,
            random_start=True,
            fixed_max_leverage=3.0,
            seed=args.seed,
        )

    vector_environment = DummyVecEnv([make_train_environment])
    vector_environment = VecNormalize(
        vector_environment,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    model = PPO(
        "MlpPolicy",
        vector_environment,
        learning_rate=3e-4,
        n_steps=1_024,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    model_path = args.output_prefix.with_suffix(".zip")
    normalizer_path = args.output_prefix.with_name(f"{args.output_prefix.name}_vecnormalize.pkl")
    model.save(str(args.output_prefix))
    vector_environment.save(str(normalizer_path))
    vector_environment.training = False
    vector_environment.norm_reward = False

    ppo_report = _evaluate(
        model,
        vector_environment,
        validation_frame,
        feature_names=features,
        episode_steps=args.episode_steps,
        episode_count=args.evaluation_episodes,
        seed=args.seed,
        policy="ppo",
        max_entries_per_window=args.max_entries_per_window,
        entry_rate_window_minutes=args.entry_rate_window_minutes,
    )
    random_report = _evaluate(
        model,
        vector_environment,
        validation_frame,
        feature_names=features,
        episode_steps=args.episode_steps,
        episode_count=args.evaluation_episodes,
        seed=args.seed,
        policy="random",
        max_entries_per_window=args.max_entries_per_window,
        entry_rate_window_minutes=args.entry_rate_window_minutes,
    )
    hold_report = _evaluate(
        model,
        vector_environment,
        validation_frame,
        feature_names=features,
        episode_steps=args.episode_steps,
        episode_count=args.evaluation_episodes,
        seed=args.seed,
        policy="hold",
        max_entries_per_window=args.max_entries_per_window,
        entry_rate_window_minutes=args.entry_rate_window_minutes,
    )
    report = {
        "experiment": "ppo-core-risk-controller-baseline-v1",
        "model_artifact": str(model_path),
        "normalizer_artifact": str(normalizer_path),
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "device": str(model.device),
        },
        "data": {
            "source": str(args.input),
            "tail_rows": len(frame),
            "train_rows": len(train_frame),
            "purge_rows": args.purge_bars,
            "validation_rows": len(validation_frame),
            "training_start": train_frame.index[0].isoformat(),
            "training_end": train_frame.index[-1].isoformat(),
            "validation_start": validation_frame.index[0].isoformat(),
            "validation_end": validation_frame.index[-1].isoformat(),
            "mark_price_available": all(name in frame for name in ("mark_open", "mark_high", "mark_low", "mark_close")),
        },
        "observation_names": list(make_train_environment().core.observation_names),
        "actions": [action.name for action in PortfolioAction],
        "risk": {
            "episode_max_leverage": 3.0,
            "base_allocation_fraction": 0.10,
            "max_drawdown": 0.20,
            "max_holding_bars": 10,
            "recovery_step": 1.25,
            "max_recovery_multiplier": 1.50,
            "max_entries_per_window": args.max_entries_per_window,
            "entry_rate_window_minutes": args.entry_rate_window_minutes,
        },
        "training": {
            "algorithm": "PPO",
            "requested_total_timesteps": args.total_timesteps,
            "actual_total_timesteps": int(model.num_timesteps),
            "episode_steps": args.episode_steps,
            "seed": args.seed,
            "learning_rate": 3e-4,
            "n_steps": 1_024,
            "batch_size": 256,
            "n_epochs": 5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "net_arch": [128, 128],
        },
        "validation": {
            "ppo": ppo_report,
            "all_hold": hold_report,
            "random_policy": random_report,
        },
        "limitations": [
            "This is a single-seed PPO smoke baseline, not a walk-forward RL result.",
            "Core data lacks Mark Price OHLC and event-complete funding, so liquidation uses documented trade-price fallback.",
            "CatBoost and macro predictions are excluded until leak-free OOF features are available.",
        ],
    }
    report_path = args.output_prefix.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
