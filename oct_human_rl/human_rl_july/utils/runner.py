"""
Shared episode / evaluation runner.

Training, evaluation and the baseline comparison all call into these two
functions, so a number reported in one place means exactly the same thing as
the same number reported elsewhere.

Note on seeding: `run_episode` resets the environment **once**, with the seed it
was given.  An earlier revision reset a second time inside the loop, which
silently discarded the seed and made evaluation unreproducible.
"""

from typing import Optional, List, Dict, Any, Sequence

from env.collaborative_env import CollaborativeTaskEnv
from agent.human_observer import HumanBehaviorObserver
from human.simulated_human import HUMAN_PROFILES
from utils.episode import EpisodeTracker
from utils.metrics import CoordinationMetrics, EpisodeStats


def profile_label(info: Dict[str, Any]) -> str:
    """Name of the partner(s) in this episode."""
    metrics = info.get("human_metrics_list") or []
    if not metrics:
        return "none"
    return "+".join(m.get("profile", "?") for m in metrics)


def run_episode(env: CollaborativeTaskEnv,
                policy,
                hbo: HumanBehaviorObserver,
                seed: Optional[int] = None,
                deterministic: bool = True,
                render: bool = False,
                collect_trace: bool = False):
    """
    Run one full episode.

    Returns (EpisodeStats, trace) where trace is a per-step list of dicts when
    `collect_trace` is set, otherwise an empty list.
    """
    obs, info = env.reset(seed=seed)
    hbo.reset()
    obs = hbo.enhance(obs)
    if hasattr(policy, "reset"):
        policy.reset()

    tracker = EpisodeTracker(env)
    trace: List[dict] = []
    terminated = truncated = False

    while not (terminated or truncated):
        action, _, _ = policy.select_action(obs, info, deterministic=deterministic)
        hbo.observe_action(action, env)

        raw_obs, reward, terminated, truncated, info = env.step(action)
        tracker.record(action, reward, info)
        hbo.update(info)
        obs = hbo.enhance(raw_obs)

        if render:
            env.render()
        if collect_trace:
            trace.append({
                "step":        info["step"],
                "action":      action,
                "worker":      tracker.worker_of(action),
                "reward":      round(reward, 3),
                "conflict":    info["conflict"],
                "num_busy":    info["num_busy"],
                "tasks_done":  info["tasks_done"],
                "orders_done": info["orders_done"],
            })

    return tracker.finish(profile_label(info), terminated), trace


def evaluate_policy(policy_or_factory,
                    profiles: Sequence[str],
                    episodes: int = 30,
                    seed: int = 0,
                    env_kwargs: Optional[dict] = None,
                    deterministic: bool = True,
                    render: bool = False,
                    metrics: Optional[CoordinationMetrics] = None,
                    observer_enabled: bool = True
                    ) -> CoordinationMetrics:
    """
    Evaluate a policy across the given human profiles.

    `policy_or_factory` may be a ready policy object (e.g. a trained PPOAgent)
    or a callable `factory(env) -> policy` for baselines that need to know the
    environment's shape. `observer_enabled` must match how the agent was
    trained, or it is handed a feature block it never learned to read.
    """
    env_kwargs = dict(env_kwargs or {})
    metrics = metrics or CoordinationMetrics(window=10 ** 9)

    for profile_name in profiles:
        env = CollaborativeTaskEnv(
            human_profile=HUMAN_PROFILES[profile_name],
            render_mode="human" if render else None,
            **env_kwargs)
        if hasattr(policy_or_factory, "select_action") and not isinstance(policy_or_factory, type):
            policy = policy_or_factory                 # a ready policy instance
        elif callable(policy_or_factory):
            policy = policy_or_factory(env)            # a factory or a class
        else:
            raise TypeError("policy_or_factory must be a policy or a callable "
                            "returning one")
        hbo = HumanBehaviorObserver(num_humans=env.num_humans,
                                    total_tasks=env.num_global_tasks,
                                    enabled=observer_enabled)
        for ep in range(episodes):
            stats, _ = run_episode(env, policy, hbo,
                                   seed=seed + ep,
                                   deterministic=deterministic,
                                   render=render)
            metrics.record(stats)
    return metrics
