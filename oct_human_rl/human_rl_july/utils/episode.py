"""
EpisodeTracker — one place where per-episode statistics are assembled.

Training, evaluation and the baseline policies all funnel through this class,
so every reported number is produced by identical logic.  Each field is read
from what the environment actually reports rather than being re-derived, which
is where the previous revision's completion and synchronisation figures went
wrong.
"""

from typing import Optional

from utils.metrics import EpisodeStats


class EpisodeTracker:
    """Accumulates one episode's statistics from (action, info) pairs."""

    def __init__(self, env):
        self.wait_action     = env.wait_action
        self.num_ai          = env.num_ai
        self.num_global_tasks = env.num_global_tasks
        self.max_steps       = env.max_steps
        self.num_orders      = env.num_orders
        self.total_tasks     = env.num_global_tasks
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        self.steps = 0
        self.total_reward = 0.0
        self.human_tasks = 0
        self.ai_tasks = 0
        self.role_switches = 0
        self.parallel_steps = 0
        self.idle_steps = 0
        self.conflict_count = 0
        self.wait_actions = 0
        self.reassignments = 0
        self._prev_kind: Optional[str] = None
        self._last_info: dict = {}

    # ------------------------------------------------------------------ #
    def worker_of(self, action: int) -> Optional[int]:
        """Worker index addressed by an action, or None for WAIT."""
        if action == self.wait_action:
            return None
        return action // self.num_global_tasks

    def is_human_action(self, action: int) -> bool:
        w = self.worker_of(action)
        return w is not None and w >= self.num_ai

    # ------------------------------------------------------------------ #
    def record(self, action: int, reward: float, info: dict):
        """Call once per env.step, with that step's action, reward and info."""
        self.steps += 1
        self.total_reward += reward
        self._last_info = info

        if info.get("conflict"):
            self.conflict_count += 1

        if action == self.wait_action:
            self.wait_actions += 1
        elif not info.get("conflict"):
            # A genuine, accepted assignment.
            if info.get("reassigned"):
                self.reassignments += 1
            kind = "human" if self.is_human_action(action) else "ai"
            if kind == "human":
                self.human_tasks += 1
            else:
                self.ai_tasks += 1
            if self._prev_kind is not None and kind != self._prev_kind:
                self.role_switches += 1
            self._prev_kind = kind

        num_busy = info.get("num_busy", 0)
        if num_busy >= 2:
            self.parallel_steps += 1
        elif num_busy == 0:
            self.idle_steps += 1

    # ------------------------------------------------------------------ #
    def finish(self, profile: str, terminated: bool) -> EpisodeStats:
        info = self._last_info
        tasks_done  = info.get("tasks_done", 0)
        orders_done = info.get("orders_done", 0)
        on_time     = info.get("orders_on_time", 0)
        solved      = bool(terminated)
        return EpisodeStats(
            profile          = profile,
            steps            = self.steps,
            total_reward     = self.total_reward,
            tasks_completed  = tasks_done,
            total_tasks      = self.total_tasks,
            orders_completed = orders_done,
            orders_on_time   = on_time,
            num_orders       = self.num_orders,
            human_tasks      = self.human_tasks,
            ai_tasks         = self.ai_tasks,
            role_switches    = self.role_switches,
            parallel_steps   = self.parallel_steps,
            idle_steps       = self.idle_steps,
            conflict_count   = self.conflict_count,
            wait_actions     = self.wait_actions,
            reassignments    = self.reassignments,
            completion_time  = float(self.steps if solved else self.max_steps),
            max_steps        = self.max_steps,
            solved           = solved,
        )
