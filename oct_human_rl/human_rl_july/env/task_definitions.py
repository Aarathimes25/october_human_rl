"""
Task and order definitions for the warehouse collaborative scenario.

Order template (one fulfilment job) — dependency graph:

      pick_item ─────┐
                     ├──▶ pack_box ──▶ label_box ──▶ weigh_box ──▶ dispatch
      fetch_box ─────┘

The environment runs several of these orders concurrently.  Each order carries
its own arrival step, deadline and priority, so the agent must decide not only
*who* does a task but *which order* to advance first.

Task times are deliberately tuned so that no single worker dominates:
  - a fast / expert human beats the AI on pick_item and fetch_box
  - the AI beats the human on packing (precision work)
  - label / weigh / dispatch are close to even
This makes the allocation decision genuinely non-trivial.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


# ── Single task within an order ───────────────────────────────────────────
@dataclass(frozen=True)
class TaskSpec:
    idx: int                        # index within the order template
    name: str
    difficulty: float               # 0.0 (trivial) .. 1.0 (very hard)
    human_time: float               # baseline human steps at speed = 1.0
    ai_time: float                  # steps the AI takes
    prerequisites: Tuple[int, ...] = ()
    description: str = ""


ORDER_TEMPLATE: List[TaskSpec] = [
    TaskSpec(idx=0, name="pick_item", difficulty=0.3,
             human_time=4.0, ai_time=3.5,
             prerequisites=(), description="Pick the item from the shelf"),

    TaskSpec(idx=1, name="fetch_box", difficulty=0.4,
             human_time=5.0, ai_time=4.0,
             prerequisites=(), description="Fetch an empty box from storage"),

    TaskSpec(idx=2, name="pack_box", difficulty=0.6,
             human_time=8.0, ai_time=5.5,
             prerequisites=(0, 1), description="Pack the item into the box"),

    TaskSpec(idx=3, name="label_box", difficulty=0.2,
             human_time=2.0, ai_time=2.0,
             prerequisites=(2,), description="Apply shipping label"),

    TaskSpec(idx=4, name="weigh_box", difficulty=0.2,
             human_time=2.0, ai_time=1.5,
             prerequisites=(3,), description="Weigh and verify shipment"),

    TaskSpec(idx=5, name="dispatch", difficulty=0.3,
             human_time=3.0, ai_time=2.5,
             prerequisites=(4,), description="Place on conveyor belt"),
]

TASKS_PER_ORDER  = len(ORDER_TEMPLATE)
TASK_NAMES       = [t.name for t in ORDER_TEMPLATE]
TOTAL_DIFFICULTY = sum(t.difficulty for t in ORDER_TEMPLATE)
MEAN_DIFFICULTY  = TOTAL_DIFFICULTY / TASKS_PER_ORDER

# Reference duration of one order if a single average worker did every task
# back to back.  Used to derive sensible deadlines.
SERIAL_HUMAN_TIME = sum(t.human_time for t in ORDER_TEMPLATE)
SERIAL_AI_TIME    = sum(t.ai_time    for t in ORDER_TEMPLATE)


# ── A live order in the environment ───────────────────────────────────────
@dataclass
class Order:
    """One fulfilment job currently on the warehouse floor."""
    order_id:     int
    slot:         int            # position in the env's concurrent-order array
    arrival_step: int
    deadline:     float          # absolute step index by which it should finish
    priority:     float          # 0.0 (routine) .. 1.0 (urgent)
    status:       List[float] = field(default_factory=list)   # per task: 0 / 0.5 / 1
    completed_at: float = -1.0

    def __post_init__(self):
        if not self.status:
            self.status = [0.0] * TASKS_PER_ORDER

    # ── queries ──────────────────────────────────────────────────────── #
    @property
    def is_complete(self) -> bool:
        return all(s == 1.0 for s in self.status)

    @property
    def tasks_done(self) -> int:
        return sum(1 for s in self.status if s == 1.0)

    def prerequisites_met(self, task_idx: int) -> bool:
        return all(self.status[p] == 1.0
                   for p in ORDER_TEMPLATE[task_idx].prerequisites)

    def is_eligible(self, task_idx: int) -> bool:
        """Pending, unassigned, and all prerequisites finished."""
        return self.status[task_idx] == 0.0 and self.prerequisites_met(task_idx)

    def slack(self, current_step: int) -> float:
        """Steps remaining before the deadline (may be negative when late)."""
        return self.deadline - current_step

    def is_late(self, current_step: int) -> bool:
        return (not self.is_complete) and current_step > self.deadline


# ── Backwards-compatible aliases ──────────────────────────────────────────
# Earlier revisions exposed a flat `WAREHOUSE_TASKS` / `NUM_TASKS` pair.
WAREHOUSE_TASKS = ORDER_TEMPLATE
NUM_TASKS       = TASKS_PER_ORDER
Task            = TaskSpec
