"""
Baseline coordination policies.

These exist so the reinforcement-learning result has something to be measured
against.  All three see exactly the same information as the RL agent (the
observation and the env's legality mask) and none of them may take an illegal
action — anything illegal degrades to WAIT, so no baseline is unfairly
penalised for conflicts.

    StaticPolicy   predefined task sequence with round-robin assignment.
                   This is the "traditional static workflow" the project
                   argues against: it never reacts to who its partner is.

    GreedyPolicy   a strong hand-written heuristic — most urgent task first,
                   given to whichever free worker is estimated to finish it
                   soonest, using only the observable speed signal.

    RandomPolicy   uniform over legal actions; the floor.

Every policy exposes  select_action(obs, info) -> int  so they are drop-in
replacements for PPOAgent in the evaluation loop.
"""

import numpy as np
from typing import Optional, List

from env.task_definitions import ORDER_TEMPLATE, TASKS_PER_ORDER


class BasePolicy:
    """Shared plumbing: legality checking and a safe WAIT fallback."""

    name = "base"

    def __init__(self, env, seed: int = 0):
        self.wait_action      = env.wait_action
        self.num_global_tasks = env.num_global_tasks
        self.num_ai           = env.num_ai
        self.num_workers      = env.num_workers
        self.num_orders       = env.num_orders
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    def reset(self):
        pass

    def action_for(self, worker_idx: int, gid: int) -> int:
        return worker_idx * self.num_global_tasks + gid

    def _legal(self, action: int, info: dict) -> bool:
        mask = info.get("action_mask")
        return bool(mask[action]) if mask is not None else True

    def _safe(self, action: Optional[int], info: dict) -> int:
        """Return the action if legal, otherwise WAIT."""
        if action is None or not self._legal(action, info):
            return self.wait_action
        return action

    # ------------------------------------------------------------------ #
    def propose(self, obs, info) -> Optional[int]:
        raise NotImplementedError

    def select_action(self, obs, info, deterministic: bool = True):
        """Mirrors PPOAgent.select_action; log-prob and value are unused."""
        return self._safe(self.propose(obs, info), info), 0.0, 0.0


class RandomPolicy(BasePolicy):
    """Randomly start available work without destructive task churn.

    WAIT and in-progress takeovers are legal environment actions, but sampling
    uniformly across *all* legal actions made this baseline spend most of an
    episode waiting or repeatedly throwing away progress.  That looks like a
    frozen dashboard and is not a useful random-allocation comparison.  The
    baseline now samples a free-worker/ready-task pair whenever one exists and
    waits only while the team is genuinely occupied or dependency-blocked.
    """

    name = "random"

    def propose(self, obs, info) -> Optional[int]:
        eligible = list(info.get("eligible_tasks", []))
        busy = info.get("worker_busy", [])
        free = [i for i in range(self.num_workers)
                if i < len(busy) and not busy[i]]
        if not eligible or not free:
            return self.wait_action

        candidates = [self.action_for(worker, gid)
                      for worker in free for gid in eligible]
        legal = [action for action in candidates if self._legal(action, info)]
        if not legal:
            return self.wait_action
        return int(self.rng.choice(legal))


class StaticPolicy(BasePolicy):
    """
    Predefined sequence, fixed round-robin assignment, zero adaptation.

    Walks the tasks in template order (order 0 first, then order 1, …) and hands
    them to workers in rotation.  If the next task in the sequence is not yet
    startable, or its designated worker is busy, the policy simply waits — it
    has no mechanism for reordering or reassigning.
    """

    name = "static"

    def __init__(self, env, seed: int = 0):
        super().__init__(env, seed)
        self.reset()

    def reset(self):
        self._cursor = 0            # position in the fixed task sequence
        self._next_worker = 0       # round-robin pointer

    def propose(self, obs, info) -> Optional[int]:
        eligible = set(info.get("eligible_tasks", []))
        busy = info.get("worker_busy", [])

        # Skip past anything already finished or in progress.
        while self._cursor < self.num_global_tasks:
            gid = self._cursor
            slot, idx = divmod(gid, TASKS_PER_ORDER)
            status = info["orders"][slot]["status"][idx]
            if status == 1.0 or status == 0.5:
                self._cursor += 1
                continue
            break

        if self._cursor >= self.num_global_tasks:
            return self.wait_action

        gid = self._cursor
        if gid not in eligible:
            return self.wait_action           # blocked: wait, never reorder

        worker = self._next_worker % self.num_workers
        if worker < len(busy) and busy[worker]:
            return self.wait_action           # its turn, but it is busy: wait

        self._next_worker += 1
        self._cursor += 1
        return self.action_for(worker, gid)


class GreedyPolicy(BasePolicy):
    """
    Urgency-first heuristic with observable-speed-based worker choice.

    1. Rank every eligible task by urgency:
         priority weight, minus remaining slack, plus a bonus for tasks that sit
         early in the dependency chain (finishing them unblocks more work).
    2. Give the top task to whichever free worker is estimated to complete it
       soonest, using only `observed_speed` — the same signal the RL agent gets.
    """

    name = "greedy"

    SLACK_WEIGHT    = 0.06
    PRIORITY_WEIGHT = 1.00
    UNBLOCK_WEIGHT  = 0.15
    MIN_SPEED       = 0.15

    def _task_score(self, gid: int, info: dict) -> float:
        slot, idx = divmod(gid, TASKS_PER_ORDER)
        order = info["orders"][slot]
        downstream = TASKS_PER_ORDER - idx        # earlier tasks unblock more
        return (self.PRIORITY_WEIGHT * order["priority"]
                - self.SLACK_WEIGHT * order["slack"]
                + self.UNBLOCK_WEIGHT * downstream)

    def _estimated_time(self, worker_idx: int, gid: int, info: dict) -> float:
        spec = ORDER_TEMPLATE[gid % TASKS_PER_ORDER]
        if worker_idx < self.num_ai:
            return spec.ai_time
        m = info["workers"][worker_idx]
        speed = max(m.get("speed", 0.5), self.MIN_SPEED)
        # Slower observed speed and higher observed error rate both inflate the
        # estimate; fatigue adds the same 60% ceiling the simulator applies.
        penalty = (1.0 + m.get("error_rate", 0.0)) * (1.0 + 0.6 * m.get("fatigue", 0.0))
        return spec.human_time / speed * penalty

    # Rescue rule: only consider taking a task off a worker when there is no
    # fresh work to start, the task is still early, and the free worker would
    # finish it from scratch sooner than the current holder will from here.
    RESCUE_MAX_PROGRESS = 0.35

    def _rescue(self, info: dict, free: List[int]) -> Optional[int]:
        best = None
        for gid in info.get("reassignable_tasks", []):
            holder = next((i for i, w in enumerate(info["workers"])
                           if w["current_task"] == gid), None)
            if holder is None:
                continue
            progress = info["workers"][holder]["progress"]
            if progress > self.RESCUE_MAX_PROGRESS:
                continue                       # too far along to be worth losing
            holder_remaining = self._estimated_time(holder, gid, info) * (1.0 - progress)
            for w in free:
                if w == holder:
                    continue
                fresh = self._estimated_time(w, gid, info)
                if fresh < holder_remaining:
                    gain = holder_remaining - fresh
                    if best is None or gain > best[0]:
                        best = (gain, w, gid)
        return None if best is None else self.action_for(best[1], best[2])

    def propose(self, obs, info) -> Optional[int]:
        eligible: List[int] = list(info.get("eligible_tasks", []))
        busy = info.get("worker_busy", [])
        free = [i for i in range(self.num_workers) if i < len(busy) and not busy[i]]
        if not free:
            return self.wait_action

        if not eligible:
            return self._rescue(info, free) or self.wait_action

        gid = max(eligible, key=lambda g: self._task_score(g, info))
        worker = min(free, key=lambda w: self._estimated_time(w, gid, info))
        return self.action_for(worker, gid)


BASELINES = {
    "random": RandomPolicy,
    "static": StaticPolicy,
    "greedy": GreedyPolicy,
}


def make_baseline(name: str, env, seed: int = 0) -> BasePolicy:
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; expected one of {sorted(BASELINES)}")
    return BASELINES[name](env, seed=seed)
