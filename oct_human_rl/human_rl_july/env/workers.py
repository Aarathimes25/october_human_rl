"""
Worker abstraction — lets the environment scale beyond one human and one robot.

Both `AIWorker` and `human.simulated_human.SimulatedHuman` implement the same
small protocol, so `CollaborativeTaskEnv` treats them uniformly:

    reset()                       clear all state
    assign(global_task_id, spec)  give the worker a task
    step(dt) -> Optional[int]     advance; return the finished task id or None
    is_busy / is_waiting          availability flags
    progress                      0..1 progress on the current task
    get_metrics() -> dict         observable signals for the UI and the observer

Adding a second robot or a third human is therefore only a matter of appending
to the worker list — no change to the reward, the mask or the policy.
"""

from typing import Optional

from env.task_definitions import TaskSpec


# Number of per-worker features written into the observation vector.
# (busy, waiting, progress, observed_speed, observed_error_rate, fatigue)
WORKER_FEATURES = 6

# The first three are scheduling state (who is free, who is mid-task, how far
# along); the last three say how well this worker is performing. The
# `blind_partner` ablation zeros only the latter.
CAPABILITY_FEATURE_SLICE = slice(3, WORKER_FEATURES)


class AIWorker:
    """
    The robotic executor.

    Deterministic: fixed per-task duration, no fatigue, no errors, no latency.
    This is deliberate — it is the stable half of the team, and the whole
    coordination problem is about fitting the variable human around it.
    """

    KIND = "ai"

    def __init__(self, name: str = "ai"):
        self.name = name
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        self.current_task: Optional[int] = None
        self._task_spec: Optional[TaskSpec] = None
        self.time_on_task: float = 0.0
        self.task_budget: float = 0.0
        self.tasks_completed: int = 0
        self.busy_steps: float = 0.0        # steps spent holding a task

    # ------------------------------------------------------------------ #
    def assign(self, global_task_id: int, spec: TaskSpec):
        self.current_task = global_task_id
        self._task_spec = spec
        self.task_budget = spec.ai_time
        self.time_on_task = 0.0

    # ------------------------------------------------------------------ #
    def release(self) -> float:
        """Give up the current task, returning the fraction of it now lost."""
        lost = self.progress
        self.current_task = None
        self._task_spec = None
        self.time_on_task = 0.0
        self.task_budget = 0.0
        return lost

    # ------------------------------------------------------------------ #
    def step(self, dt: float = 1.0) -> Optional[int]:
        if self.current_task is None:
            return None
        self.busy_steps += dt
        self.time_on_task += dt
        if self.time_on_task >= self.task_budget:
            done_id = self.current_task
            self.tasks_completed += 1
            self.current_task = None
            self._task_spec = None
            return done_id
        return None

    # ── observable signals ───────────────────────────────────────────── #
    @property
    def is_busy(self) -> bool:
        return self.current_task is not None

    @property
    def is_waiting(self) -> bool:
        return False            # the robot never hesitates

    @property
    def observed_speed(self) -> float:
        return 1.0

    @property
    def observed_error_rate(self) -> float:
        return 0.0

    @property
    def fatigue(self) -> float:
        return 0.0

    @property
    def progress(self) -> float:
        if self.current_task is None or self.task_budget <= 0.0:
            return 0.0
        return min(self.time_on_task / self.task_budget, 1.0)

    def get_metrics(self) -> dict:
        return {
            "kind":         self.KIND,
            "name":         self.name,
            "profile":      "deterministic",
            "fatigue":      0.0,
            "speed":        1.0,
            "error_rate":   0.0,
            "tasks_done":   self.tasks_completed,
            "errors":       0,
            "busy":         self.is_busy,
            "waiting":      False,
            "progress":     round(self.progress, 3),
            "busy_steps":   round(self.busy_steps, 1),
            "current_task": self.current_task,
        }


def worker_features(worker) -> list:
    """The fixed-length observation block for any worker. Order matters."""
    return [
        1.0 if worker.is_busy else 0.0,
        1.0 if worker.is_waiting else 0.0,
        worker.progress,
        worker.observed_speed,
        worker.observed_error_rate,
        worker.fatigue,
    ]
