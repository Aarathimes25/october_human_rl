"""
HumanBehaviorObserver — the Human Interaction Abstraction Layer.

Maintains a rolling-window estimate of each human partner's:
  - smoothed speed        relative to baseline
  - smoothed error rate   recent unreliability
  - completion ratio      tasks finished vs tasks assigned
  - speed trend           positive = speeding up, negative = tiring
  - workload              tasks finished, normalised

These are **generalised interaction signals**, not a behavioural profile: the
agent never learns "this is a novice", only "this partner is currently slow and
error-prone".  That abstraction is what allows the policy to transfer to a
partner type it has never met.

The features are appended to the raw environment observation.

Usage
    obs, info = env.reset()
    hbo = HumanBehaviorObserver(num_humans=env.num_humans, window=10)
    obs = hbo.enhance(obs)
    ...
    hbo.observe_action(action, env)      # records assignments to humans
    obs2, r, term, trunc, info = env.step(action)
    hbo.update(info)
    obs2 = hbo.enhance(obs2)
"""

import numpy as np
from collections import deque
from typing import Dict, Any, List


# Features contributed per human partner.
FEATURES_PER_HUMAN = 5


def extra_obs_dim(num_humans: int = 1) -> int:
    return num_humans * FEATURES_PER_HUMAN


# Convenience constant for the single-human default configuration.
EXTRA_OBS_DIM = extra_obs_dim(1)


class _SingleHumanTracker:
    """Rolling statistics for one human partner."""

    def __init__(self, window: int):
        self.window = window
        self.reset()

    def reset(self):
        self._speed_hist = deque(maxlen=self.window)
        self._error_hist = deque(maxlen=self.window)
        self._tasks_assigned = 0
        self._tasks_completed = 0
        self._prev_done = 0

    def update(self, metrics: Dict[str, Any]):
        if not metrics:
            return
        self._speed_hist.append(metrics.get("speed", 0.5))
        self._error_hist.append(metrics.get("error_rate", 0.0))
        done = metrics.get("tasks_done", 0)
        if done > self._prev_done:
            self._tasks_completed += done - self._prev_done
        self._prev_done = done

    def record_assignment(self):
        self._tasks_assigned += 1

    # ── derived features ─────────────────────────────────────────────── #
    @property
    def smooth_speed(self) -> float:
        return float(np.mean(self._speed_hist)) if self._speed_hist else 0.5

    @property
    def smooth_error_rate(self) -> float:
        return float(np.mean(self._error_hist)) if self._error_hist else 0.0

    @property
    def completion_ratio(self) -> float:
        if self._tasks_assigned == 0:
            return 1.0
        return min(self._tasks_completed / self._tasks_assigned, 1.0)

    @property
    def speed_trend(self) -> float:
        """Positive = speeding up, negative = slowing down."""
        if len(self._speed_hist) < 4:
            return 0.0
        vals = list(self._speed_hist)
        half = len(vals) // 2
        return float(np.clip(np.mean(vals[half:]) - np.mean(vals[:half]), -1.0, 1.0))

    def features(self, total_tasks: int) -> List[float]:
        return [
            self.smooth_speed,
            self.smooth_error_rate,
            self.completion_ratio,
            self.speed_trend,
            min(self._tasks_completed / max(total_tasks, 1), 1.0),
        ]

    def summary(self) -> str:
        return (f"speed={self.smooth_speed:.2f} err={self.smooth_error_rate:.2f} "
                f"compl={self.completion_ratio:.2f} trend={self.speed_trend:+.2f}")

    def snapshot(self, total_tasks: int) -> Dict[str, Any]:
        """Public, serialisable view of the features emitted by this tracker."""
        features = self.features(total_tasks)
        return {
            "smooth_speed":       features[0],
            "smooth_error_rate":  features[1],
            "completion_ratio":   features[2],
            "speed_trend":        features[3],
            "completed_workload": features[4],
            "tasks_assigned":     self._tasks_assigned,
            "tasks_completed":    self._tasks_completed,
            "observations":       len(self._speed_hist),
        }


class HumanBehaviorObserver:
    """Stateful observer producing an enhanced observation vector."""

    def __init__(self, num_humans: int = 1, window: int = 10,
                 total_tasks: int = 12, enabled: bool = True):
        """enabled=False blanks the feature block without changing its width,
        so the ablation isolates the information rather than the network size."""
        self.num_humans = num_humans
        self.window = window
        self.total_tasks = total_tasks
        self.enabled = enabled
        self.trackers = [_SingleHumanTracker(window) for _ in range(num_humans)]

    @property
    def extra_dim(self) -> int:
        return extra_obs_dim(self.num_humans)

    def reset(self):
        for t in self.trackers:
            t.reset()

    # ── update from env info ─────────────────────────────────────────── #
    def update(self, info: Dict[str, Any]):
        metrics_list = info.get("human_metrics_list")
        if metrics_list is None:
            single = info.get("human_metrics")
            metrics_list = [single] if single else []
        for tracker, metrics in zip(self.trackers, metrics_list):
            tracker.update(metrics)

    def record_assignment(self, human_index: int = 0):
        """Call once each time the agent assigns a task to that human."""
        if 0 <= human_index < self.num_humans:
            self.trackers[human_index].record_assignment()

    def observe_action(self, action: int, env) -> None:
        """
        Record an assignment straight from the action, so callers never have to
        re-derive which worker was addressed.  WAIT and AI actions are ignored.
        """
        if action == env.wait_action:
            return
        w_idx = action // env.num_global_tasks
        if w_idx >= env.num_ai:                     # humans follow the AI workers
            self.record_assignment(w_idx - env.num_ai)

    # ── combine with the raw env obs ─────────────────────────────────── #
    def extra_features(self) -> np.ndarray:
        if not self.enabled:
            return np.zeros(self.extra_dim, dtype=np.float32)
        feats: List[float] = []
        for tracker in self.trackers:
            feats.extend(tracker.features(self.total_tasks))
        return np.asarray(feats, dtype=np.float32)

    def enhance(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(obs, dtype=np.float32),
                               self.extra_features()], axis=-1)

    # ── readable summary ─────────────────────────────────────────────── #
    def summary(self) -> str:
        return " | ".join(f"h{i}: {t.summary()}" for i, t in enumerate(self.trackers))

    def snapshots(self) -> List[Dict[str, Any]]:
        """Feature values currently appended to the policy observation."""
        return [tracker.snapshot(self.total_tasks) for tracker in self.trackers]
