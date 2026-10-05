"""
CollaborativeTaskEnv — custom Gymnasium environment for dynamic task
sequencing with zero-shot human–AI coordination.

Scales to any number of workers and any number of concurrent orders.
Default configuration: 1 AI robot + 1 human, 2 concurrent orders.

Global task id
──────────────
    gid = order_slot * TASKS_PER_ORDER + task_index

Action space — Discrete(NUM_WORKERS * NUM_GLOBAL_TASKS + 1)
    action <  NUM_WORKERS * NUM_GLOBAL_TASKS
        worker = action // NUM_GLOBAL_TASKS
        task   = action %  NUM_GLOBAL_TASKS
    action == NUM_WORKERS * NUM_GLOBAL_TASKS
        WAIT — let work in progress continue.  Always legal, never penalised.

Observation (flat float32 vector)
    per global task (TASK_FEATURES = 6)
        status            0 pending / 0.5 in-progress / 1 done
        difficulty        normalised task difficulty
        assignee          0 nobody / 0.5 human / 1 AI
        priority          urgency of the owning order
        slack             (deadline - step) / MAX_EPISODE_STEPS, clipped [-1,1]
        arrived           1 once the owning order has arrived
    per worker (WORKER_FEATURES = 6)
        busy, waiting, progress, observed_speed, observed_error_rate, fatigue
    global (GLOBAL_FEATURES = 3)
        time_ratio, fraction_tasks_done, fraction_orders_done

Reward
    -0.02   per step                          time pressure
    +10 x difficulty                          per completed task
    +20 x (1 + priority)                      per order finished on time
    +20 x 0.4                                 per order finished late
    -0.25 x priority per step                 per order past its deadline
    -0.30 per step per idle worker            ONLY while eligible work exists
    +0.40 per step per extra busy worker      parallelism bonus
    -1.0                                      illegal action (masked out in practice)
    +25                                       all orders complete

The idle penalty is conditional on there being work the worker could actually
take.  Waiting while the team is blocked is free — this is what makes the WAIT
action usable and is the key difference from the earlier revision, where the
agent was punished every step for situations it could not control.
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Sequence

from env.task_definitions import (ORDER_TEMPLATE, TASKS_PER_ORDER, Order,
                                  SERIAL_AI_TIME, TaskSpec)
from env.workers import (AIWorker, WORKER_FEATURES, worker_features,
                         CAPABILITY_FEATURE_SLICE)
from human.simulated_human import (SimulatedHuman, HumanProfile,
                                   HUMAN_PROFILES, TRAIN_PROFILES)

# ── Episode / scenario defaults ───────────────────────────────────────── #
MAX_EPISODE_STEPS = 70
DEFAULT_NUM_ORDERS = 2
DEFAULT_NUM_HUMANS = 1
DEFAULT_NUM_AI     = 1
ORDER_ARRIVAL_GAP  = 5        # steps between successive order arrivals

# Deadline = arrival + factor * SERIAL_AI_TIME, where factor shrinks as the
# order's priority rises — urgent orders get tighter deadlines.
DEADLINE_FACTOR_BASE  = 2.6
DEADLINE_FACTOR_SLOPE = 1.2
PRIORITY_RANGE = (0.2, 1.0)

# ── Reward constants ──────────────────────────────────────────────────── #
TIME_PENALTY      = -0.02
TASK_REWARD_SCALE = 10.0
ORDER_BONUS       = 20.0
LATE_ORDER_FACTOR = 0.4       # multiplier on ORDER_BONUS when finished late
LATE_PENALTY      = -0.25     # per step, per late order, scaled by priority
IDLE_PENALTY      = -0.30     # per step, per avoidably-idle worker
SYNC_BONUS        = 0.40      # per step, per busy worker beyond the first
INVALID_PENALTY   = -1.0
ALL_DONE_BONUS    = 25.0
REASSIGN_PENALTY  = -2.0      # scaled by the progress thrown away

# ── Observation layout ────────────────────────────────────────────────── #
TASK_FEATURES   = 6
GLOBAL_FEATURES = 3


def obs_dim_for(num_orders: int = DEFAULT_NUM_ORDERS,
                num_workers: int = DEFAULT_NUM_AI + DEFAULT_NUM_HUMANS) -> int:
    return (num_orders * TASKS_PER_ORDER * TASK_FEATURES
            + num_workers * WORKER_FEATURES
            + GLOBAL_FEATURES)


def action_dim_for(num_orders: int = DEFAULT_NUM_ORDERS,
                   num_workers: int = DEFAULT_NUM_AI + DEFAULT_NUM_HUMANS) -> int:
    return num_workers * num_orders * TASKS_PER_ORDER + 1


# Convenience constants for the default configuration.
OBS_DIM    = obs_dim_for()
ACTION_DIM = action_dim_for()


class CollaborativeTaskEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 human_profile: Optional[HumanProfile] = None,
                 human_profiles: Optional[List[HumanProfile]] = None,
                 num_orders: int = DEFAULT_NUM_ORDERS,
                 num_humans: int = DEFAULT_NUM_HUMANS,
                 num_ai: int = DEFAULT_NUM_AI,
                 arrival_gap: int = ORDER_ARRIVAL_GAP,
                 max_steps: int = MAX_EPISODE_STEPS,
                 train_profiles: Optional[List[str]] = None,
                 allow_reassignment: bool = True,
                 blind_partner: bool = False,
                 order_priorities: Optional[Sequence[float]] = None,
                 dt: float = 1.0,
                 render_mode: Optional[str] = None):
        """
        human_profile      fixed profile for every human (evaluation)
        human_profiles     per-human fixed profiles; overrides human_profile
        train_profiles     pool sampled from when no fixed profile is given;
                           defaults to TRAIN_PROFILES so held-out profiles stay unseen
        allow_reassignment when True the coordinator may take an in-progress
                           task off one worker and hand it to a free one,
                           forfeiting the progress made so far
        blind_partner      ablation: zero the per-worker capability signals so
                           the agent sees who is free but not how well they work
        order_priorities   fixed urgency per order slot instead of sampling it.
                           Deadlines tighten with priority, so pinning this
                           pins the whole time pressure of the scenario.
        """
        super().__init__()
        self.dt = dt
        self.render_mode = render_mode
        self.allow_reassignment = allow_reassignment
        self.blind_partner = blind_partner
        if order_priorities is not None:
            order_priorities = [float(np.clip(p, 0.0, 1.0))
                                for p in order_priorities]
            if not order_priorities:
                raise ValueError("order_priorities must not be empty")
        self.order_priorities = order_priorities
        self.num_orders  = num_orders
        self.num_humans  = num_humans
        self.num_ai      = num_ai
        self.num_workers = num_ai + num_humans
        self.arrival_gap = arrival_gap
        self.max_steps   = max_steps
        self.train_profiles = list(train_profiles or TRAIN_PROFILES)

        if human_profiles is not None and len(human_profiles) != num_humans:
            raise ValueError("len(human_profiles) must equal num_humans")
        self.human_profile  = human_profile
        self.human_profiles = human_profiles

        self.num_global_tasks = num_orders * TASKS_PER_ORDER
        self.obs_dim    = obs_dim_for(num_orders, self.num_workers)
        self.action_dim = action_dim_for(num_orders, self.num_workers)
        self.wait_action = self.action_dim - 1

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.action_dim)

        self.workers: List = []
        self.orders:  List[Order] = []
        self._assignee = np.full(self.num_global_tasks, -1, dtype=np.int64)
        self._step_count = 0
        self._last_action_valid = True
        self._last_action_reassign = False
        self._reassignments = 0

    # ── id helpers ────────────────────────────────────────────────────── #
    def gid(self, slot: int, task_idx: int) -> int:
        return slot * TASKS_PER_ORDER + task_idx

    def split_gid(self, gid: int) -> Tuple[int, int]:
        return divmod(gid, TASKS_PER_ORDER)

    def spec_for(self, gid: int) -> TaskSpec:
        return ORDER_TEMPLATE[gid % TASKS_PER_ORDER]

    @property
    def humans(self) -> List[SimulatedHuman]:
        return [w for w in self.workers if w.KIND == "human"]

    @property
    def human(self):
        """First human — convenience for the single-human default setup."""
        hs = self.humans
        return hs[0] if hs else None

    # ── eligibility ───────────────────────────────────────────────────── #
    def _order_arrived(self, order: Order) -> bool:
        return self._step_count >= order.arrival_step

    def _eligible_tasks(self) -> List[int]:
        """Global ids of tasks that are pending, unblocked, arrived, unassigned."""
        out = []
        for order in self.orders:
            if not self._order_arrived(order) or order.is_complete:
                continue
            for idx in range(TASKS_PER_ORDER):
                if order.is_eligible(idx):
                    out.append(self.gid(order.slot, idx))
        return out

    def _reassignable_tasks(self) -> List[int]:
        """Global ids of in-progress tasks that could be taken off their worker."""
        if not self.allow_reassignment:
            return []
        return [gid for gid in range(self.num_global_tasks)
                if self._assignee[gid] >= 0]

    def _can_assign(self, w_idx: int, gid: int) -> bool:
        """May worker w take task gid right now?

        Both `legal_action_mask` and `step` go through here, so the mask the
        policy sees and the check the environment applies cannot drift apart.
        """
        worker = self.workers[w_idx]
        if worker.is_busy:
            return False
        slot, task_idx = self.split_gid(gid)
        order = self.orders[slot]
        if not self._order_arrived(order) or not order.prerequisites_met(task_idx):
            return False

        status = order.status[task_idx]
        if status == 0.0:
            return True                                   # start it fresh
        if status == 0.5 and self.allow_reassignment:
            holder = int(self._assignee[gid])
            return holder >= 0 and holder != w_idx        # take it off someone else
        return False

    def legal_action_mask(self) -> np.ndarray:
        """
        Canonical legality, owned by the environment.

        The agent reads this straight out of `info`, so the policy's mask and
        the environment's validity check can never disagree.
        """
        mask = np.zeros(self.action_dim, dtype=bool)
        mask[self.wait_action] = True              # waiting is always allowed
        for w_idx, worker in enumerate(self.workers):
            if worker.is_busy:
                continue
            base = w_idx * self.num_global_tasks
            for gid in range(self.num_global_tasks):
                if self._can_assign(w_idx, gid):
                    mask[base + gid] = True
        return mask

    # ── observation ───────────────────────────────────────────────────── #
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        p = 0
        for order in self.orders:
            arrived = 1.0 if self._order_arrived(order) else 0.0
            slack = np.clip(order.slack(self._step_count) / self.max_steps, -1.0, 1.0)
            for idx in range(TASKS_PER_ORDER):
                spec = ORDER_TEMPLATE[idx]
                gid = self.gid(order.slot, idx)
                w_idx = self._assignee[gid]
                if w_idx < 0:
                    assignee = 0.0
                else:
                    assignee = 1.0 if self.workers[w_idx].KIND == "ai" else 0.5
                obs[p:p + TASK_FEATURES] = (
                    order.status[idx], spec.difficulty, assignee,
                    order.priority, slack, arrived,
                )
                p += TASK_FEATURES

        for worker in self.workers:
            block = obs[p:p + WORKER_FEATURES]
            block[:] = worker_features(worker)
            if self.blind_partner:
                block[CAPABILITY_FEATURE_SLICE] = 0.0
            p += WORKER_FEATURES

        obs[p]     = self._step_count / self.max_steps
        obs[p + 1] = self._tasks_done() / self.num_global_tasks
        obs[p + 2] = self._orders_done() / self.num_orders
        return obs

    # ── counters ──────────────────────────────────────────────────────── #
    def _tasks_done(self) -> int:
        return sum(o.tasks_done for o in self.orders)

    def _orders_done(self) -> int:
        return sum(1 for o in self.orders if o.is_complete)

    def _orders_on_time(self) -> int:
        return sum(1 for o in self.orders
                   if o.is_complete and o.completed_at <= o.deadline)

    def _orders_late(self) -> int:
        return sum(1 for o in self.orders
                   if o.is_complete and o.completed_at > o.deadline)

    def _num_busy(self) -> int:
        return sum(1 for w in self.workers if w.is_busy)

    # ── info ──────────────────────────────────────────────────────────── #
    def _get_info(self) -> Dict[str, Any]:
        human_metrics = [w.get_metrics() for w in self.workers if w.KIND == "human"]
        return {
            "step":            self._step_count,
            "action_mask":     self.legal_action_mask(),
            "eligible_tasks":  self._eligible_tasks(),
            "reassignable_tasks": self._reassignable_tasks(),
            "reassignments":   self._reassignments,
            "reassigned":      self._last_action_reassign,
            "workers":         [w.get_metrics() for w in self.workers],
            "worker_busy":     [w.is_busy for w in self.workers],
            "num_busy":        self._num_busy(),
            "tasks_done":      self._tasks_done(),
            "total_tasks":     self.num_global_tasks,
            "orders_done":     self._orders_done(),
            "orders_on_time":  self._orders_on_time(),
            "orders_late":     self._orders_late(),
            "num_orders":      self.num_orders,
            "conflict":        not self._last_action_valid,
            "human_metrics_list": human_metrics,
            # single-human convenience view used by the observer and the UI
            "human_metrics":   human_metrics[0] if human_metrics else {},
            "human_busy":      bool(human_metrics and human_metrics[0]["busy"]),
            "orders": [
                {
                    "order_id":  o.order_id,
                    "slot":      o.slot,
                    "priority":  round(o.priority, 3),
                    "deadline":  round(o.deadline, 1),
                    "arrival":   o.arrival_step,
                    "arrived":   self._order_arrived(o),
                    "complete":  o.is_complete,
                    "late":      o.is_late(self._step_count),
                    "slack":     round(o.slack(self._step_count), 1),
                    "tasks_done": o.tasks_done,
                    "status":    list(o.status),
                }
                for o in self.orders
            ],
        }

    # ── reset ─────────────────────────────────────────────────────────── #
    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # The training curriculum names the partner for the episode; without an
        # override we sample the pool as usual.
        forced = None
        if options and options.get("profile"):
            name = options["profile"]
            if name not in HUMAN_PROFILES:
                raise ValueError(f"unknown profile {name!r}")
            forced = HUMAN_PROFILES[name]

        # ── build the workers ─────────────────────────────────────────── #
        self.workers = [AIWorker(name=f"ai_{i}") for i in range(self.num_ai)]
        for h in range(self.num_humans):
            if forced is not None:
                profile = forced
            elif self.human_profiles is not None:
                profile = self.human_profiles[h]
            elif self.human_profile is not None:
                profile = self.human_profile
            else:
                # Sample ONLY from the training pool so held-out profiles
                # remain genuinely unseen.
                profile = HUMAN_PROFILES[str(self.np_random.choice(self.train_profiles))]
            self.workers.append(SimulatedHuman(
                profile=profile,
                seed=int(self.np_random.integers(0, 1_000_000)),
                name=f"human_{h}"))

        # ── build the order queue ─────────────────────────────────────── #
        self.orders = []
        for slot in range(self.num_orders):
            if self.order_priorities is None:
                priority = float(self.np_random.uniform(*PRIORITY_RANGE))
            else:
                priority = self.order_priorities[slot % len(self.order_priorities)]
            arrival  = slot * self.arrival_gap
            factor   = DEADLINE_FACTOR_BASE - DEADLINE_FACTOR_SLOPE * priority
            self.orders.append(Order(
                order_id=slot,
                slot=slot,
                arrival_step=arrival,
                deadline=arrival + factor * SERIAL_AI_TIME,
                priority=priority,
            ))

        self._assignee = np.full(self.num_global_tasks, -1, dtype=np.int64)
        self._step_count = 0
        self._last_action_valid = True
        self._last_action_reassign = False
        self._reassignments = 0

        return self._get_obs(), self._get_info()

    # ── step ──────────────────────────────────────────────────────────── #
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        if not 0 <= action < self.action_dim:
            raise ValueError(f"action {action} outside [0, {self.action_dim})")

        reward = TIME_PENALTY
        self._last_action_valid = True
        self._last_action_reassign = False

        # ── 1. apply the assignment ───────────────────────────────────── #
        if action != self.wait_action:
            w_idx = action // self.num_global_tasks
            gid   = action %  self.num_global_tasks
            slot, task_idx = self.split_gid(gid)
            order  = self.orders[slot]
            worker = self.workers[w_idx]

            if self._can_assign(w_idx, gid):
                holder = int(self._assignee[gid])
                if holder >= 0:                      # taking it off someone else
                    reward += REASSIGN_PENALTY * self.workers[holder].release()
                    self._reassignments += 1
                    self._last_action_reassign = True
                order.status[task_idx] = 0.5
                self._assignee[gid] = w_idx
                worker.assign(gid, ORDER_TEMPLATE[task_idx])
            else:
                self._last_action_valid = False
                reward += INVALID_PENALTY

        # ── 2. advance every worker ───────────────────────────────────── #
        for worker in self.workers:
            done_gid = worker.step(dt=self.dt)
            if done_gid is None:
                continue
            slot, task_idx = self.split_gid(done_gid)
            self.orders[slot].status[task_idx] = 1.0
            self._assignee[done_gid] = -1
            reward += TASK_REWARD_SCALE * ORDER_TEMPLATE[task_idx].difficulty

        self._step_count += 1

        # ── 3. order completion bonuses ───────────────────────────────── #
        for order in self.orders:
            if order.is_complete and order.completed_at < 0:
                order.completed_at = float(self._step_count)
                if order.completed_at <= order.deadline:
                    reward += ORDER_BONUS * (1.0 + order.priority)
                else:
                    reward += ORDER_BONUS * LATE_ORDER_FACTOR

        # ── 4. lateness pressure on unfinished orders ─────────────────── #
        for order in self.orders:
            if order.is_late(self._step_count) and self._order_arrived(order):
                reward += LATE_PENALTY * order.priority

        # ── 5. parallelism bonus ──────────────────────────────────────── #
        num_busy = self._num_busy()
        if num_busy > 1:
            reward += SYNC_BONUS * (num_busy - 1)

        # ── 6. idle penalty — only when work was actually available ───── #
        # Evaluated after the action, so a worker assigned this step is not
        # punished, and nobody is punished while the team is genuinely blocked.
        if self._eligible_tasks():
            idle = sum(1 for w in self.workers if not w.is_busy)
            reward += IDLE_PENALTY * idle

        # ── 7. termination ────────────────────────────────────────────── #
        all_done   = all(o.is_complete for o in self.orders)
        terminated = all_done
        truncated  = (not all_done) and self._step_count >= self.max_steps
        if all_done:
            reward += ALL_DONE_BONUS

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    # ── render ────────────────────────────────────────────────────────── #
    def render(self):
        icons = {0.0: "[ ]", 0.5: "[~]", 1.0: "[x]"}
        print(f"\n── Step {self._step_count:3d} "
              f"({self._tasks_done()}/{self.num_global_tasks} tasks, "
              f"{self._orders_done()}/{self.num_orders} orders) ──")
        for order in self.orders:
            tag = "waiting" if not self._order_arrived(order) else (
                  "DONE" if order.is_complete else
                  f"slack={order.slack(self._step_count):+.0f}")
            print(f"  Order {order.order_id} "
                  f"(prio={order.priority:.2f}, due={order.deadline:.0f}, {tag})")
            row = "  ".join(
                f"{icons[order.status[i]]}{ORDER_TEMPLATE[i].name}"
                for i in range(TASKS_PER_ORDER))
            print(f"    {row}")
        for w in self.workers:
            m = w.get_metrics()
            state = ("waiting" if m["waiting"] else
                     f"task {m['current_task']} {m['progress']*100:3.0f}%"
                     if m["busy"] else "idle")
            print(f"  {m['name']:<9} [{m['profile']:<13}] {state:<18} "
                  f"spd={m['speed']:.2f} ftg={m['fatigue']:.2f} err={m['error_rate']:.2f}")
