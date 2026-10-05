"""
SimulatedHuman — models a human warehouse worker.

A profile is described by five independent behavioural axes:

  speed             multiplier on base task time  (0.45 expert … 2.4 novice)
  error_rate        per-step probability of an error that forces a redo
  fatigue_rate      fatigue gained per step of work; fatigue slows the worker
  response_latency  steps between being given a task and actually starting it
                    (models "delayed responses")
  speed_jitter      per-task random variation in speed (models erratic workers)
  strategy          how difficulty biases the worker's pace:
                      "sequential"  neutral
                      "easy_first"  quick on easy tasks, slow on hard ones
                      "hard_first"  quick on hard tasks, slow on easy ones

Every one of these axes is *observable only through its consequences* — the
agent never sees the profile, just the signals in HumanBehaviorObserver.
That is what makes the zero-shot claim meaningful.
"""

import numpy as np
from typing import Optional

from env.task_definitions import ORDER_TEMPLATE, TaskSpec, MEAN_DIFFICULTY


# ── Strategy model ────────────────────────────────────────────────────────
# A worker's pace is scaled by  1 + coeff * (difficulty - mean_difficulty).
# easy_first  → coeff > 0 : hard tasks take proportionally longer
# hard_first  → coeff < 0 : easy tasks take proportionally longer
STRATEGY_COEFF = {
    "sequential": 0.0,
    "easy_first": +1.5,
    "hard_first": -1.5,
}
STRATEGY_BIAS_CLIP = (0.5, 2.0)

# Per-task random speed variation is clipped to this band so an "erratic"
# worker stays erratic without occasionally stalling the whole episode.
JITTER_CLIP = (0.55, 1.80)


class HumanProfile:
    """Defines a human 'type' — used for zero-shot generalisation."""

    def __init__(self,
                 name: str,
                 speed: float,
                 error_rate: float,
                 fatigue_rate: float,
                 response_latency: float = 0.0,
                 speed_jitter: float = 0.0,
                 strategy: str = "sequential"):
        if strategy not in STRATEGY_COEFF:
            raise ValueError(f"unknown strategy {strategy!r}; "
                             f"expected one of {sorted(STRATEGY_COEFF)}")
        self.name = name
        self.speed = speed                        # lower = faster
        self.error_rate = error_rate              # 0–1 per step
        self.fatigue_rate = fatigue_rate          # fatigue gained per worked step
        self.response_latency = response_latency  # steps before work begins
        self.speed_jitter = speed_jitter          # std-dev of per-task speed noise
        self.strategy = strategy

    def difficulty_bias(self, difficulty: float) -> float:
        """Pace multiplier this worker applies to a task of the given difficulty."""
        coeff = STRATEGY_COEFF[self.strategy]
        bias = 1.0 + coeff * (difficulty - MEAN_DIFFICULTY)
        return float(np.clip(bias, *STRATEGY_BIAS_CLIP))

    def __repr__(self):
        return (f"HumanProfile({self.name}, spd={self.speed:.2f}, "
                f"err={self.error_rate:.2f}, lat={self.response_latency:.1f}, "
                f"jit={self.speed_jitter:.2f}, {self.strategy})")


# ── Human profile library ─────────────────────────────────────────────────
# speed < 1.0 → faster than baseline;  speed > 1.0 → slower than baseline
HUMAN_PROFILES = {
    # ── training profiles ────────────────────────────────────────────────
    # These vary along speed / error / fatigue / latency, all with the
    # neutral "sequential" strategy.
    "average": HumanProfile("average", speed=1.00, error_rate=0.05,
                            fatigue_rate=0.010, response_latency=0.0,
                            speed_jitter=0.10),
    "fast":    HumanProfile("fast",    speed=0.65, error_rate=0.07,
                            fatigue_rate=0.012, response_latency=0.0,
                            speed_jitter=0.15),
    "slow":    HumanProfile("slow",    speed=1.60, error_rate=0.04,
                            fatigue_rate=0.008, response_latency=1.0,
                            speed_jitter=0.10),
    "expert":  HumanProfile("expert",  speed=0.45, error_rate=0.01,
                            fatigue_rate=0.003, response_latency=0.0,
                            speed_jitter=0.05),
    "novice":  HumanProfile("novice",  speed=2.40, error_rate=0.28,
                            fatigue_rate=0.030, response_latency=2.0,
                            speed_jitter=0.25),

    # ── held-out zero-shot test profiles (never seen during training) ────
    # Each one varies along an axis or combination the agent never trained on.
    "tired":      HumanProfile("tired",      speed=1.40, error_rate=0.14,
                               fatigue_rate=0.028, response_latency=1.0,
                               speed_jitter=0.18),
    "easy_first": HumanProfile("easy_first", speed=1.00, error_rate=0.05,
                               fatigue_rate=0.010, response_latency=0.0,
                               speed_jitter=0.10, strategy="easy_first"),
    "hard_first": HumanProfile("hard_first", speed=1.00, error_rate=0.05,
                               fatigue_rate=0.010, response_latency=0.0,
                               speed_jitter=0.10, strategy="hard_first"),
    "hesitant":   HumanProfile("hesitant",   speed=0.90, error_rate=0.06,
                               fatigue_rate=0.012, response_latency=3.0,
                               speed_jitter=0.12),
    "erratic":    HumanProfile("erratic",    speed=1.10, error_rate=0.20,
                               fatigue_rate=0.020, response_latency=1.0,
                               speed_jitter=0.40),
}

# ── Zero-shot train / test split ─────────────────────────────────────────
# The agent is trained ONLY on TRAIN_PROFILES.  TEST_PROFILES are held out
# entirely, so evaluating on them is a legitimate zero-shot claim.
TRAIN_PROFILES = ["average", "fast", "slow", "expert", "novice"]
TEST_PROFILES  = ["tired", "easy_first", "hard_first", "hesitant", "erratic"]

assert not (set(TRAIN_PROFILES) & set(TEST_PROFILES)), "train/test profiles overlap"
assert set(TRAIN_PROFILES) | set(TEST_PROFILES) == set(HUMAN_PROFILES), \
    "every profile must be assigned to exactly one split"


class SimulatedHuman:
    """
    A simulated human collaborator working one task at a time.

    Lifecycle of an assigned task:
        assign()  →  [latency phase]  →  [work phase]  →  done

    Fatigue accumulates while working and slows the worker by up to 60%.
    Errors add redo time to the remaining budget.
    """

    KIND = "human"
    FATIGUE_SLOWDOWN = 0.6      # at full fatigue the worker is 60% slower
    REDO_FRACTION    = 0.4      # an error costs 40% of the task's base time

    def __init__(self, profile: Optional[HumanProfile] = None, seed: int = 42,
                 name: str = "human"):
        self.profile = profile or HUMAN_PROFILES["average"]
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        self.fatigue: float = 0.0
        self.current_task: Optional[int] = None      # global task id
        self._task_spec: Optional[TaskSpec] = None
        self.time_on_task: float = 0.0
        self.task_budget: float = 0.0
        self.latency_left: float = 0.0
        self.tasks_completed: int = 0
        self.total_errors: int = 0
        self.busy_steps: float = 0.0        # steps spent holding a task
        self.recent_speeds: list = []
        self.recent_errors: list = []

    # ------------------------------------------------------------------ #
    def assign(self, global_task_id: int, spec: TaskSpec):
        """Assign a new task. Work begins after the worker's response latency."""
        fatigue_penalty = 1.0 + self.fatigue * self.FATIGUE_SLOWDOWN
        jitter = 1.0
        if self.profile.speed_jitter > 0.0:
            jitter = float(np.clip(
                self.rng.normal(1.0, self.profile.speed_jitter), *JITTER_CLIP))
        bias = self.profile.difficulty_bias(spec.difficulty)

        self.task_budget = (spec.human_time * self.profile.speed
                            * fatigue_penalty * jitter * bias)
        self.time_on_task = 0.0
        self.latency_left = self.profile.response_latency
        self.current_task = global_task_id
        self._task_spec = spec

    # ------------------------------------------------------------------ #
    def release(self) -> float:
        """Give up the current task, returning the fraction of it now lost.

        Fatigue and the error history stay: the worker really did that effort,
        and the coordinator should still see how they were performing.
        """
        lost = self.progress
        self.current_task = None
        self._task_spec = None
        self.time_on_task = 0.0
        self.task_budget = 0.0
        self.latency_left = 0.0
        return lost

    # ------------------------------------------------------------------ #
    def step(self, dt: float = 1.0) -> Optional[int]:
        """
        Advance the worker by dt.
        Returns the global task id if a task was completed this tick, else None.
        """
        if self.current_task is None:
            return None
        self.busy_steps += dt      # counted from the moment the task is held

        # ── response-latency phase: the worker has not started yet ────── #
        if self.latency_left > 0.0:
            self.latency_left = max(0.0, self.latency_left - dt)
            return None

        self.time_on_task += dt
        self.fatigue = min(1.0, self.fatigue + self.profile.fatigue_rate * dt)

        # ── random error: adds redo time to the remaining budget ──────── #
        if self.rng.random() < self.profile.error_rate * dt:
            redo = self._task_spec.human_time * self.profile.speed * self.REDO_FRACTION
            self.task_budget += redo
            self.total_errors += 1
            self.recent_errors.append(1)
        else:
            self.recent_errors.append(0)
        if len(self.recent_errors) > 20:
            self.recent_errors.pop(0)

        # ── completion ────────────────────────────────────────────────── #
        if self.time_on_task >= self.task_budget:
            ratio = self._task_spec.human_time / max(self.time_on_task, 0.01)
            self.recent_speeds.append(float(min(ratio, 1.0)))
            if len(self.recent_speeds) > 10:
                self.recent_speeds.pop(0)
            self.tasks_completed += 1
            done_id = self.current_task
            self.current_task = None
            self._task_spec = None
            return done_id

        return None

    # ── observable signals ───────────────────────────────────────────── #
    @property
    def observed_speed(self) -> float:
        """Normalised speed estimate in [0,1]; 1 = at or above baseline pace."""
        if not self.recent_speeds:
            return 0.5
        return float(np.clip(np.mean(self.recent_speeds), 0.0, 1.0))

    @property
    def observed_error_rate(self) -> float:
        if not self.recent_errors:
            return 0.0
        return float(np.mean(self.recent_errors))

    @property
    def is_busy(self) -> bool:
        return self.current_task is not None

    @property
    def is_waiting(self) -> bool:
        """Assigned a task but not yet started (response latency)."""
        return self.current_task is not None and self.latency_left > 0.0

    @property
    def progress(self) -> float:
        if self.current_task is None or self.task_budget <= 0.0:
            return 0.0
        return float(np.clip(self.time_on_task / self.task_budget, 0.0, 1.0))

    def get_metrics(self) -> dict:
        return {
            "kind":        self.KIND,
            "name":        self.name,
            "profile":     self.profile.name,
            "fatigue":     round(self.fatigue, 3),
            "speed":       round(self.observed_speed, 3),
            "error_rate":  round(self.observed_error_rate, 3),
            "tasks_done":  self.tasks_completed,
            "errors":      self.total_errors,
            "busy":        self.is_busy,
            "waiting":     self.is_waiting,
            "progress":    round(self.progress, 3),
            "busy_steps":  round(self.busy_steps, 1),
            "current_task": self.current_task,
        }
