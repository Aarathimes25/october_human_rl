"""
Coordination metrics and the Performance Analytics data model.

Tracks per-episode statistics and derives:
  1. Coordination Efficiency Score  (CES)
  2. Role switch rate
  3. Human–AI synchronisation rate
  4. On-time order rate
  5. Zero-shot generalisation gap, reported as *seen* vs *held-out*

Every quantity below is computed from values the environment actually reports
(`info["tasks_done"]`, `info["orders_on_time"]`, `info["num_busy"]`, …) rather
than being inferred or assumed.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Sequence
import numpy as np


@dataclass
class EpisodeStats:
    profile:          str
    steps:            int
    total_reward:     float
    tasks_completed:  int
    total_tasks:      int
    orders_completed: int
    orders_on_time:   int
    num_orders:       int
    human_tasks:      int      # assignments made to humans
    ai_tasks:         int      # assignments made to the AI
    role_switches:    int
    parallel_steps:   int      # steps on which 2+ workers were busy
    idle_steps:       int      # steps on which no worker was busy
    conflict_count:   int      # illegal actions attempted
    wait_actions:     int
    reassignments:    int      # in-progress tasks pulled off a worker
    completion_time:  float    # steps to finish everything, or max_steps
    max_steps:        int
    solved:           bool     # all orders completed before truncation

    def as_dict(self) -> dict:
        return asdict(self)


class CoordinationMetrics:
    """
    Accumulates episode stats and computes the coordination efficiency score.

    Coordination Efficiency Score (CES)
    ───────────────────────────────────
      CES = 0.35 * completion_rate
          + 0.20 * speed_score
          + 0.15 * sync_score
          + 0.20 * on_time_rate
          - 0.10 * conflict_rate

      completion_rate = tasks_completed / total_tasks
      speed_score     = 1 - completion_time / max_steps
      sync_score      = parallel_steps / steps
      on_time_rate    = orders_on_time / num_orders
      conflict_rate   = conflict_count / steps

    CES lies in [-0.10, 0.90]; higher is better.
    """

    WEIGHTS = dict(completion=0.35, speed=0.20, sync=0.15,
                   on_time=0.20, conflict=0.10)

    def __init__(self, window: int = 50):
        self.window = window
        self.history: List[EpisodeStats] = []

    # ── recording ────────────────────────────────────────────────────── #
    def record(self, stats: EpisodeStats):
        self.history.append(stats)

    def reset(self):
        self.history.clear()

    # ── per-episode score ────────────────────────────────────────────── #
    @classmethod
    def ces(cls, s: EpisodeStats) -> float:
        w = cls.WEIGHTS
        completion_rate = s.tasks_completed / max(s.total_tasks, 1)
        speed_score     = max(1.0 - s.completion_time / max(s.max_steps, 1), 0.0)
        sync_score      = s.parallel_steps / max(s.steps, 1)
        on_time_rate    = s.orders_on_time / max(s.num_orders, 1)
        conflict_rate   = s.conflict_count / max(s.steps, 1)
        return (w["completion"] * completion_rate
                + w["speed"]    * speed_score
                + w["sync"]     * sync_score
                + w["on_time"]  * on_time_rate
                - w["conflict"] * conflict_rate)

    # ── windows ──────────────────────────────────────────────────────── #
    def recent(self, n: Optional[int] = None) -> List[EpisodeStats]:
        return self.history[-(n or self.window):]

    def for_profile(self, profile: str) -> List[EpisodeStats]:
        return [s for s in self.history if s.profile == profile]

    # ── aggregates ───────────────────────────────────────────────────── #
    def _mean(self, fn, hist) -> float:
        return float(np.mean([fn(s) for s in hist])) if hist else 0.0

    def mean_ces(self, n=None) -> float:
        return self._mean(self.ces, self.recent(n))

    def mean_reward(self, n=None) -> float:
        return self._mean(lambda s: s.total_reward, self.recent(n))

    def mean_completion(self, n=None) -> float:
        return self._mean(lambda s: s.tasks_completed / max(s.total_tasks, 1),
                          self.recent(n))

    def solve_rate(self, n=None) -> float:
        return self._mean(lambda s: float(s.solved), self.recent(n))

    def mean_steps(self, n=None) -> float:
        return self._mean(lambda s: s.steps, self.recent(n))

    def on_time_rate(self, n=None) -> float:
        return self._mean(lambda s: s.orders_on_time / max(s.num_orders, 1),
                          self.recent(n))

    def sync_rate(self, n=None) -> float:
        return self._mean(lambda s: s.parallel_steps / max(s.steps, 1),
                          self.recent(n))

    def role_switch_rate(self, n=None) -> float:
        return self._mean(lambda s: s.role_switches / max(s.steps, 1),
                          self.recent(n))

    def conflict_rate(self, n=None) -> float:
        return self._mean(lambda s: s.conflict_count / max(s.steps, 1),
                          self.recent(n))

    def mean_reassignments(self, n=None) -> float:
        return self._mean(lambda s: float(s.reassignments), self.recent(n))

    def human_share(self, n=None) -> float:
        """Fraction of assignments given to a human partner."""
        hist = self.recent(n)
        if not hist:
            return 0.0
        h = sum(s.human_tasks for s in hist)
        a = sum(s.ai_tasks for s in hist)
        return h / max(h + a, 1)

    # ── per-profile view ─────────────────────────────────────────────── #
    def profile_table(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for profile in sorted({s.profile for s in self.history}):
            hist = self.for_profile(profile)
            out[profile] = {
                "episodes":     len(hist),
                "ces":          self._mean(self.ces, hist),
                "reward":       self._mean(lambda s: s.total_reward, hist),
                "steps":        self._mean(lambda s: s.steps, hist),
                "completion":   self._mean(
                    lambda s: s.tasks_completed / max(s.total_tasks, 1), hist),
                "solve_rate":   self._mean(lambda s: float(s.solved), hist),
                "on_time":      self._mean(
                    lambda s: s.orders_on_time / max(s.num_orders, 1), hist),
                "sync":         self._mean(
                    lambda s: s.parallel_steps / max(s.steps, 1), hist),
                "conflicts":    self._mean(lambda s: float(s.conflict_count), hist),
                "reassignments": self._mean(lambda s: float(s.reassignments), hist),
                "human_share":  (
                    sum(s.human_tasks for s in hist)
                    / max(sum(s.human_tasks + s.ai_tasks for s in hist), 1)),
            }
        return out

    # ── zero-shot generalisation ─────────────────────────────────────── #
    def zsc_report(self,
                   seen_profiles: Sequence[str],
                   held_out_profiles: Sequence[str]) -> Dict[str, object]:
        """
        Split performance into training ('seen') and held-out ('zero-shot')
        partner groups and report the generalisation gap between them.

        A small gap means the policy coordinates with an unfamiliar partner
        about as well as with a familiar one — the claim the project makes.
        """
        table = self.profile_table()
        seen = {p: table[p] for p in seen_profiles if p in table}
        held = {p: table[p] for p in held_out_profiles if p in table}

        def avg(group, key):
            return float(np.mean([v[key] for v in group.values()])) if group else 0.0

        seen_ces, held_ces = avg(seen, "ces"), avg(held, "ces")
        return {
            "seen":            seen,
            "held_out":        held,
            "seen_ces":        seen_ces,
            "held_out_ces":    held_ces,
            "zsc_gap":         seen_ces - held_ces,
            "relative_gap":    (seen_ces - held_ces) / seen_ces if seen_ces > 1e-9 else 0.0,
            "seen_solve":      avg(seen, "solve_rate"),
            "held_out_solve":  avg(held, "solve_rate"),
            "seen_reward":     avg(seen, "reward"),
            "held_out_reward": avg(held, "reward"),
        }

    # ── printable summaries ───────────────────────────────────────────── #
    def summary(self, n: Optional[int] = None) -> str:
        return (f"Episodes={len(self.recent(n))}  "
                f"CES={self.mean_ces(n):.3f}  "
                f"Reward={self.mean_reward(n):+.1f}  "
                f"Completion={self.mean_completion(n)*100:.1f}%  "
                f"Solved={self.solve_rate(n)*100:.1f}%  "
                f"OnTime={self.on_time_rate(n)*100:.1f}%  "
                f"Sync={self.sync_rate(n):.3f}  "
                f"RoleSwitch={self.role_switch_rate(n):.3f}")

    def profile_summary(self) -> str:
        table = self.profile_table()
        head = (f"{'profile':<12}{'eps':>5}{'CES':>8}{'reward':>9}{'steps':>7}"
                f"{'compl':>8}{'solved':>8}{'onTime':>8}{'sync':>7}{'human%':>8}"
                f"{'reasgn':>8}")
        lines = [head, "-" * len(head)]
        for p, v in table.items():
            lines.append(
                f"{p:<12}{v['episodes']:>5.0f}{v['ces']:>8.3f}{v['reward']:>9.1f}"
                f"{v['steps']:>7.1f}{v['completion']*100:>7.1f}%"
                f"{v['solve_rate']*100:>7.1f}%{v['on_time']*100:>7.1f}%"
                f"{v['sync']:>7.3f}{v['human_share']*100:>7.1f}%"
                f"{v['reassignments']:>8.2f}")
        return "\n".join(lines)

    # ── export ────────────────────────────────────────────────────────── #
    def to_records(self) -> List[dict]:
        out = []
        for s in self.history:
            rec = s.as_dict()
            rec["ces"] = self.ces(s)
            out.append(rec)
        return out
