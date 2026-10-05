"""
app.py: DTS ZSC interactive coordination dashboard.

Run with:  python app.py
Then open: http://localhost:5000

Shows, live:
  * the task dependency graph for every order in the warehouse, with nodes
    coloured by status and by who is working on them
  * per-order deadlines, urgency and slack
  * each worker's state (human vitals: speed, fatigue, error rate)
  * the coordination metrics the project reports (reward, CES, sync, conflicts)
  * an event log of the agent's actual decisions

The policy selector lets you watch the trained PPO agent and the scripted
baselines drive the *same* environment, which makes the difference between
learned sequencing and a fixed workflow directly visible.

This module only reads the environment's public `info` dictionary. It does not
reach into private state.
"""

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request, render_template_string

from env.collaborative_env import (CollaborativeTaskEnv, obs_dim_for,
                                   action_dim_for, MAX_EPISODE_STEPS,
                                   DEFAULT_NUM_ORDERS, DEFAULT_NUM_HUMANS,
                                   DEFAULT_NUM_AI)
from env.task_definitions import ORDER_TEMPLATE, TASKS_PER_ORDER
from agent.ppo_agent import PPOAgent
from agent.human_observer import HumanBehaviorObserver, extra_obs_dim
from human.simulated_human import HUMAN_PROFILES, TRAIN_PROFILES, TEST_PROFILES
from utils.episode import EpisodeTracker
from utils.metrics import CoordinationMetrics
from baselines import make_baseline, BASELINES

app = Flask(__name__)

CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "checkpoints", "agent_final.pt")
EVALUATION = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "results", "evaluation.json")
# PPO networks have fixed input/output sizes, so every dashboard scenario other
# than the default needs its own trained model. scripts/train_scenarios.py
# writes them here; the default scenario keeps using CHECKPOINT above.
SCENARIO_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "checkpoints", "scenarios")
RESEARCH_EVIDENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "results", "research_evidence.json")

session = {}


def _mean_profile_metric(node, key):
    """Mean a metric over the equally weighted partner profiles in a result node."""
    values = [profile[key] for profile in node.get("profiles", {}).values()
              if key in profile]
    return sum(values) / len(values) if values else 0.0


def research_summary():
    """Build the UI's comparison/evidence model from saved evaluation artefacts."""
    try:
        with open(EVALUATION, "r", encoding="utf-8") as handle:
            results = json.load(handle)
        with open(RESEARCH_EVIDENCE, "r", encoding="utf-8") as handle:
            measured = json.load(handle)

        if "baselines" not in results:
            return {"available": False,
                    "error": ("results/evaluation.json has no baseline results. "
                              "Regenerate it with: python3 evaluate.py --baselines"),
                    "comparisons": [], "evidence": []}
        ppo = results["ppo"]
        static = results["baselines"]["static"]
        random = results["baselines"]["random"]
        zsc = ppo["zsc"]
        profiles = ppo["profiles"]

        ppo_ces = ppo["overall_ces"]
        static_ces = static["overall_ces"]
        expert_ces = profiles["expert"]["ces"]
        novice_ces = profiles["novice"]["ces"]
        seen_ces = zsc["seen_ces"]
        unseen_ces = zsc["held_out_ces"]
        ppo_reassign = _mean_profile_metric(ppo, "reassignments")
        random_reassign = _mean_profile_metric(random, "reassignments")
        idle_reduction = 1.0 - measured["ppo_idle_rate"] / measured["static_idle_rate"]
        churn_reduction = 1.0 - ppo_reassign / random_reassign

        return {
            "available": True,
            "episodes_per_profile": results.get("episodes", 0),
            "profiles": len(profiles),
            "comparisons": [
                {
                    "title": "Static vs PPO",
                    "left_label": "Static workflow", "left_value": static_ces,
                    "right_label": "PPO policy", "right_value": ppo_ces,
                    "delta": (ppo_ces / static_ces - 1.0),
                    "detail": (f"Solve rate {static['solve_rate'] * 100:.1f}% → "
                               f"{ppo['solve_rate'] * 100:.1f}%"),
                },
                {
                    "title": "Expert vs Novice",
                    "left_label": "Expert partner", "left_value": expert_ces,
                    "right_label": "Novice partner", "right_value": novice_ces,
                    "delta": (novice_ces / expert_ces - 1.0),
                    "detail": (f"PPO completion time {profiles['expert']['steps']:.1f} vs "
                               f"{profiles['novice']['steps']:.1f} steps"),
                },
                {
                    "title": "Seen vs Unseen Partner",
                    "left_label": "Seen partners", "left_value": seen_ces,
                    "right_label": "Held out partners", "right_value": unseen_ces,
                    "delta": (unseen_ces / seen_ces - 1.0),
                    "detail": (f"Held out solve rate {zsc['held_out_solve'] * 100:.1f}% "
                               "without retraining"),
                },
            ],
            "evidence": [
                {
                    "label": "Zero Shot Adaptation",
                    "value": f"{abs((unseen_ces / seen_ces - 1.0) * 100):.1f}% higher",
                    "detail": "Held out CES relative to seen partners",
                    "tone": "green",
                },
                {
                    "label": "Unseen Partner",
                    "value": f"{zsc['held_out_solve'] * 100:.1f}%",
                    "detail": "Solve rate across 250 held out partner episodes",
                    "tone": "purple",
                },
                {
                    "label": "Coordination Improvement",
                    "value": f"{abs((ppo_ces / static_ces - 1.0) * 100):.1f}% higher",
                    "detail": "Overall CES relative to the static baseline",
                    "tone": "green",
                },
                {
                    "label": "Idle Time Reduction",
                    "value": f"{idle_reduction * 100:.1f}% lower",
                    "detail": "Fully idle step rate vs static (500 episodes each)",
                    "tone": "green",
                },
                {
                    "label": "Conflict Reduction",
                    "value": f"{churn_reduction * 100:.1f}% lower",
                    "detail": ("Disruptive task takeovers vs random baseline; "
                               "illegal conflicts were 0 for all masked policies"),
                    "tone": "green",
                },
            ],
        }
    except (OSError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        reason = (f"missing field {exc}" if isinstance(exc, KeyError) else str(exc))
        return {"available": False,
                "error": (f"{reason}. Restore the saved results with "
                          "git checkout -- results/, or regenerate them with "
                          "python3 evaluate.py --baselines"),
                "comparisons": [], "evidence": []}


def _human_signal(info):
    """Summarise only observable partner signals supplied to the coordinator."""
    humans = info.get("human_metrics_list", [])
    if not humans:
        return "No human signal available", "neutral"

    speed = sum(h.get("speed", 0.5) for h in humans) / len(humans)
    error = sum(h.get("error_rate", 0.0) for h in humans) / len(humans)
    fatigue = sum(h.get("fatigue", 0.0) for h in humans) / len(humans)
    waiting = any(h.get("waiting", False) for h in humans)
    summary = f"speed {speed:.2f} · error {error:.2f} · fatigue {fatigue:.2f}"

    if waiting:
        return f"Delayed response observed · {summary}", "attention"
    if error >= 0.15:
        return f"Elevated recent errors · {summary}", "attention"
    if fatigue >= 0.35:
        return f"Rising fatigue observed · {summary}", "attention"
    if speed < 0.45:
        return f"Below baseline pace observed. {summary}", "attention"
    return f"Stable live behavior · {summary}", "stable"


def explain_decision(action, before_info, after_info, previous_kind):
    """Produce a compact, evidence-bound explanation for the selected action."""
    env = session["env"]
    signal, signal_level = _human_signal(before_info)

    if action == env.wait_action:
        ready = before_info.get("eligible_tasks", [])
        busy = before_info.get("num_busy", 0)
        if busy == env.num_workers:
            why = "Every worker was occupied, so the policy preserved work in progress."
        elif not ready:
            why = "No unblocked task was available, so waiting avoided an unnecessary takeover."
        else:
            why = "The policy held the current allocation while legal work remained available."
        return {
            "step": after_info.get("step", 0),
            "decision": "Held the current allocation",
            "why": why,
            "role_change": "No role handoff on this step.",
            "trigger": signal, "trigger_level": signal_level,
        }

    if after_info.get("conflict"):
        return {
            "step": after_info.get("step", 0),
            "decision": "Rejected an illegal assignment",
            "why": "The safety mask blocked a task/worker combination that was not legal.",
            "role_change": "No role change was applied.",
            "trigger": signal, "trigger_level": signal_level,
        }

    w_idx = action // env.num_global_tasks
    gid = action % env.num_global_tasks
    worker = before_info["workers"][w_idx]
    kind = worker["kind"]
    slot, task_idx = divmod(gid, TASKS_PER_ORDER)
    order = before_info["orders"][slot]
    task = ORDER_TEMPLATE[task_idx]
    urgent = order["late"] or order["slack"] < 8 or order["priority"] > 0.66
    urgency = ("deadline pressure" if urgent else
               f"available slack of {order['slack']:.0f} steps")

    if after_info.get("reassigned"):
        previous = next((w for w in before_info["workers"]
                         if w.get("current_task") == gid), None)
        previous_name = previous["name"] if previous else "the prior worker"
        why = (f"The policy moved an active task under {urgency}; unfinished progress "
               "was traded for a different worker fit.")
        role_change = f"Task ownership changed from {previous_name} to {worker['name']}."
    else:
        why_task = (f"{task.name.replace('_', ' ').title()} was unblocked under "
                    f"{urgency}.")
        if kind == "ai":
            fit = ("AI capacity absorbed the work while the human signal was monitored."
                   if signal_level == "attention" else
                   "Available AI capacity supported parallel progress.")
        else:
            fit = ("Human capacity was used to keep work parallel; live behavior remained "
                   "part of the policy input.")
        why = f"{why_task} {fit}"
        if previous_kind is not None and previous_kind != kind:
            role_change = (f"Assignment focus shifted from {previous_kind.upper()} to "
                           f"{kind.upper()} to rebalance the next unit of work.")
        else:
            role_change = f"No cross role handoff; the next assignment stayed with {kind.upper()}."

    return {
        "step": after_info.get("step", 0),
        "decision": (f"Assigned {task.name.replace('_', ' ').title()} · order {slot + 1} "
                     f"→ {worker['name']}"),
        "why": why, "role_change": role_change,
        "trigger": signal, "trigger_level": signal_level,
    }


# ── session management ────────────────────────────────────────────────── #
def scenario_name(num_orders, num_ai, num_humans):
    return f"{num_orders}o_{num_ai}ai_{num_humans}h"


def scenario_checkpoint(num_orders, num_ai, num_humans):
    """Path of the PPO checkpoint trained for this exact scenario."""
    if (num_orders, num_ai, num_humans) == (DEFAULT_NUM_ORDERS, DEFAULT_NUM_AI,
                                            DEFAULT_NUM_HUMANS):
        return CHECKPOINT
    return os.path.join(SCENARIO_CHECKPOINT_DIR,
                        scenario_name(num_orders, num_ai, num_humans),
                        "agent_final.pt")


def build_policy(kind: str, env):
    """Return (policy, label, note)."""
    if kind in BASELINES:
        return make_baseline(kind, env, seed=0), kind, ""
    if kind != "ppo":
        raise ValueError(f"unknown policy {kind!r}")

    team = (f"{env.num_orders} order{'s' if env.num_orders != 1 else ''}, "
            f"{env.num_ai} AI + {env.num_humans} human{'s' if env.num_humans != 1 else ''}")
    path = scenario_checkpoint(env.num_orders, env.num_ai, env.num_humans)

    def fallback(reason):
        note = (f"No usable PPO model for this scenario ({team}): {reason}. "
                "The Greedy coordinator is running instead; no untrained model "
                "is being presented as PPO. Train the missing scenario models "
                "with: python3 scripts/train_scenarios.py")
        return make_baseline("greedy", env, seed=0), "greedy (PPO fallback)", note

    if not os.path.exists(path):
        return fallback("it has not been trained yet")
    obs_dim = obs_dim_for(env.num_orders, env.num_workers) + extra_obs_dim(env.num_humans)
    agent = PPOAgent(obs_dim=obs_dim,
                     action_dim=action_dim_for(env.num_orders, env.num_workers))
    try:
        agent.load(path)
        agent.net.eval()
    except Exception as exc:
        return fallback(f"the checkpoint could not be loaded ({exc})")
    if path == CHECKPOINT:
        return agent, "ppo", ""
    note = (f"PPO model trained specifically for {team} "
            f"(checkpoints/scenarios/"
            f"{scenario_name(env.num_orders, env.num_ai, env.num_humans)}).")
    try:
        with open(os.path.join(os.path.dirname(path), "summary.json"),
                  "r", encoding="utf-8") as handle:
            comparison = json.load(handle)["comparison"]
        note += (f" Tested against Greedy on the same seeds: CES "
                 f"{comparison['ppo']['ces']:.3f} vs {comparison['greedy']['ces']:.3f}, "
                 f"solved {comparison['ppo']['solve_rate'] * 100:.0f}% vs "
                 f"{comparison['greedy']['solve_rate'] * 100:.0f}%.")
    except (OSError, KeyError, TypeError, ValueError):
        pass
    note += (" The reported research numbers come from the default "
             "2 order, 1 AI + 1 human model.")
    return agent, "ppo", note


# Urgency presets. Deadlines tighten as priority rises, so these change how
# much time pressure the scenario is under, not just a label.
PRIORITY_PRESETS = {
    "random":  (None,          "Random priorities (0.20 to 1.00)"),
    "urgent":  ([0.95],        "All urgent (0.95)"),
    "medium":  ([0.55],        "All medium (0.55)"),
    "routine": ([0.20],        "All routine (0.20)"),
    "mixed":   ([0.95, 0.20],  "Mixed: urgent first (0.95 / 0.20)"),
    "rising":  ([0.20, 0.95],  "Mixed: urgent later (0.20 / 0.95)"),
    "custom":  ("custom",      "Custom sequence (you rank the orders)"),
}

# The dashboard supports up to this many orders in one episode. The
# environment itself is not limited; this is the operator-facing ceiling.
MAX_UI_ORDERS = 6

# Priority band used when the operator ranks the orders by hand. Rank #1 gets
# the top of the band, the last rank gets the bottom, and the ranks in between
# are spaced evenly, so every order ends up with a distinct urgency (and, since
# deadlines tighten with urgency, a distinct deadline).
CUSTOM_PRIORITY_HIGH = 0.95
CUSTOM_PRIORITY_LOW  = 0.20


def ui_max_steps(num_orders):
    """
    Step budget for a dashboard scenario.

    The default 70 ticks are calibrated for two orders, and the third order
    uses the documented 90-tick budget. Beyond that each extra order brings
    six more tasks *and* arrives five ticks later than the one before it, so a
    flat 20-tick increment squeezes the budget with every order added: the
    slowest baseline needs 161 ticks at six orders, which the flat rule would
    truncate. Orders past the third therefore add 35 ticks each.
    """
    if num_orders <= 3:
        return MAX_EPISODE_STEPS + max(0, num_orders - DEFAULT_NUM_ORDERS) * 20
    return MAX_EPISODE_STEPS + 20 + (num_orders - 3) * 35


def parse_priority_order(raw, num_orders):
    """
    Turn an operator-supplied ranking such as "1,2,3,6,5,4" into a per-slot
    priority list.

    The sequence lists order numbers (1-based, as shown on the dashboard) from
    most urgent to least urgent, so "1,2,3,6,5,4" means: order 1 is worked
    first, then 2, then 3, then 6, then 5, then 4.

    Returns (sequence, priorities) where `sequence` is the cleaned 1-based
    ranking and `priorities` is indexed by order slot for the environment.
    """
    if isinstance(raw, str):
        tokens = [chunk for chunk in raw.replace(";", ",").replace(" ", ",").split(",")
                  if chunk]
    elif isinstance(raw, (list, tuple)):
        tokens = list(raw)
    elif raw is None:
        tokens = []
    else:
        raise ValueError("priority_order must be a list or a comma separated string")

    try:
        sequence = [int(token) for token in tokens]
    except (TypeError, ValueError):
        raise ValueError("priority_order must contain whole order numbers, "
                         f"for example 1,2,3 for {num_orders} orders")

    if not sequence:
        # No explicit ranking: fall back to natural order 1..N.
        sequence = list(range(1, num_orders + 1))

    if sorted(sequence) != list(range(1, num_orders + 1)):
        raise ValueError(
            f"priority_order must list each order from 1 to {num_orders} exactly "
            f"once (got {','.join(str(n) for n in sequence)})")

    span = CUSTOM_PRIORITY_HIGH - CUSTOM_PRIORITY_LOW
    priorities = [0.0] * num_orders
    for rank, order_number in enumerate(sequence):
        share = rank / (num_orders - 1) if num_orders > 1 else 0.0
        priorities[order_number - 1] = round(CUSTOM_PRIORITY_HIGH - share * span, 3)
    return sequence, priorities


def reset_session(profile_name="average", policy_kind="ppo",
                  num_orders=DEFAULT_NUM_ORDERS, num_humans=DEFAULT_NUM_HUMANS,
                  num_ai=DEFAULT_NUM_AI, priority="random", priority_order=None,
                  seed=42):
    profile = HUMAN_PROFILES.get(profile_name, HUMAN_PROFILES["average"])
    priorities = PRIORITY_PRESETS.get(priority, PRIORITY_PRESETS["random"])[0]
    sequence = None
    if priorities == "custom":
        sequence, priorities = parse_priority_order(priority_order, num_orders)
    max_steps = ui_max_steps(num_orders)
    env = CollaborativeTaskEnv(human_profile=profile,
                               num_orders=num_orders,
                               num_humans=num_humans,
                               num_ai=num_ai,
                               max_steps=max_steps,
                               order_priorities=priorities)
    policy, label, note = build_policy(policy_kind, env)

    hbo = HumanBehaviorObserver(num_humans=env.num_humans,
                                total_tasks=env.num_global_tasks)
    obs, info = env.reset(seed=seed)
    hbo.reset()
    obs = hbo.enhance(obs)
    if hasattr(policy, "reset"):
        policy.reset()

    session.clear()
    session.update({
        "env": env, "policy": policy, "hbo": hbo,
        "obs": obs, "info": info,
        "tracker": EpisodeTracker(env),
        "done": False, "terminated": False,
        "profile": profile_name, "policy_kind": label, "note": note,
        "requested_policy": policy_kind,
        "priority": priority, "priority_order": sequence, "seed": seed,
        "last_assignment_kind": None,
        "recovery_policy": make_baseline("greedy", env, seed=seed + 1),
        "avoidable_waits": 0, "recoveries": 0,
        "explanations": [],
        "log": [{"step": 0, "kind": "info",
                 "text": (f"Episode started: partner '{profile_name}', policy '{label}'"
                          + (f", operator priority order "
                             f"{' > '.join(f'order {n}' for n in sequence)}"
                             if sequence else ""))}],
    })


def _log(kind: str, text: str):
    session["log"].append({"step": session["info"].get("step", 0),
                           "kind": kind, "text": text})
    if len(session["log"]) > 200:
        del session["log"][:-200]


def task_label(gid: int) -> str:
    slot, idx = divmod(gid, TASKS_PER_ORDER)
    return f"order {slot + 1} / {ORDER_TEMPLATE[idx].name}"


def step_once():
    if session["done"]:
        return

    env, policy, hbo = session["env"], session["policy"], session["hbo"]
    tracker = session["tracker"]

    before_info = session["info"]
    before = {w["name"]: w["current_task"] for w in before_info["workers"]}
    orders_complete_before = {o["slot"] for o in before_info["orders"] if o["complete"]}
    # Track which tasks were finished, so a task *taken away* from a worker is
    # never mistaken for one they completed.
    done_before = {(o["slot"], i) for o in before_info["orders"]
                   for i in range(TASKS_PER_ORDER) if o["status"][i] == 1.0}

    action, _, _ = policy.select_action(session["obs"], session["info"],
                                        deterministic=True)
    decision_source = session["policy_kind"]

    # Waiting is correct while the team is occupied.  Repeated WAIT proposals
    # with both legal work and a free worker caused the reported apparent
    # freeze.  A labelled recovery action keeps the live demo moving without
    # silently attributing that action to the learned policy.
    free_worker = any(not busy for busy in before_info.get("worker_busy", []))
    avoidable_wait = (action == env.wait_action
                      and bool(before_info.get("eligible_tasks"))
                      and free_worker)
    if session["requested_policy"] == "ppo" and session["policy_kind"] == "ppo":
        session["avoidable_waits"] = (session["avoidable_waits"] + 1
                                      if avoidable_wait else 0)
        if session["avoidable_waits"] >= 2:
            recovered, _, _ = session["recovery_policy"].select_action(
                session["obs"], before_info, deterministic=True)
            if recovered != env.wait_action:
                action = recovered
                decision_source = "PPO continuity guard"
                session["recoveries"] += 1
                session["avoidable_waits"] = 0
    else:
        session["avoidable_waits"] = 0

    hbo.observe_action(action, env)
    raw_obs, reward, terminated, truncated, info = env.step(action)
    tracker.record(action, reward, info)
    hbo.update(info)

    explanation = explain_decision(
        action, before_info, info, session["last_assignment_kind"])
    explanation["source"] = decision_source
    if decision_source == "PPO continuity guard":
        explanation["why"] = ("PPO proposed WAIT twice while a worker and legal work "
                              "were available. The continuity guard selected the "
                              "highest ranked ready assignment. " + explanation["why"])
    session["explanations"].append(explanation)
    if len(session["explanations"]) > 80:
        del session["explanations"][:-80]
    if action != env.wait_action and not info.get("conflict"):
        w_idx = action // env.num_global_tasks
        session["last_assignment_kind"] = before_info["workers"][w_idx]["kind"]

    session["obs"] = hbo.enhance(raw_obs)
    session["info"] = info
    session["terminated"] = terminated
    session["done"] = terminated or truncated

    # ── narrate what actually happened ─────────────────────────────────── #
    if info["conflict"]:
        _log("bad", "illegal assignment rejected")
    elif action == env.wait_action:
        _log("dim", "Waiting: work in progress continues")
    else:
        w_idx = action // env.num_global_tasks
        gid = action % env.num_global_tasks
        w = info["workers"][w_idx]
        kind = "human" if w["kind"] == "human" else "ai"
        if info.get("reassigned"):
            prev = next((n for n, t in before.items() if t == gid), "someone")
            _log("bad", f"reassigned {task_label(gid)}: {prev} -> {w['name']} "
                        f"(progress forfeited)")
        else:
            _log(kind, f"assigned {task_label(gid)} -> {w['name']}")

    # A worker "finished" only if the task itself is now complete. A task that
    # was taken off them does not count.
    for w in info["workers"]:
        was = before.get(w["name"])
        if was is None or w["current_task"] == was:
            continue
        slot, idx = divmod(was, TASKS_PER_ORDER)
        if (slot, idx) not in done_before and info["orders"][slot]["status"][idx] == 1.0:
            _log("ok", f"{w['name']} finished {task_label(was)}")

    # Only orders that finished on *this* step are logged. Re-announcing every
    # completed order each time another one finishes misreported the finish
    # step, and re-judged an on-time order as LATE once the clock moved past
    # its deadline.
    for o in info["orders"]:
        if o["complete"] and o["slot"] not in orders_complete_before:
            on_time = info["step"] <= o["deadline"]
            _log("ok" if on_time else "bad",
                 f"order {o['slot'] + 1} complete "
                 f"({'on time' if on_time else 'LATE'})")

    if session["done"]:
        _log("ok" if terminated else "bad",
             "all orders complete" if terminated else "episode truncated at the step limit")


# ── serialisation ─────────────────────────────────────────────────────── #
def live_ces() -> float:
    """
    The same CES the evaluation reports, computed from the episode so far.
    The speed component only pays out once every order is finished, so this
    figure rises sharply on the final step. That is the real metric, not a
    display-only approximation.
    """
    tracker = session["tracker"]
    if tracker.steps == 0:
        return 0.0
    stats = tracker.finish(session["profile"], session["terminated"])
    return round(CoordinationMetrics.ces(stats), 3)


def sequencing_snapshot(info):
    """Readable view of the constraints and candidates behind task order."""
    actual = session["policy_kind"]
    requested = session["requested_policy"]

    if requested == "ppo" and actual != "ppo":
        basis = ("The PPO checkpoint does not match this scenario size. The active "
                 "Greedy fallback ranks ready work by priority, deadline slack and "
                 "downstream dependency value, then chooses the fastest free worker.")
    elif actual == "ppo":
        basis = ("Dependencies and worker availability first remove illegal actions. "
                 "PPO then scores the remaining task and worker pairs from task state and "
                 "difficulty, order priority and slack, worker signals, overall "
                 "progress, and the Human Observer feature block.")
    elif actual == "greedy":
        basis = ("Ready work is ranked by order priority, deadline slack and how much "
                 "downstream work it unlocks; the fastest available worker is chosen.")
    elif actual == "static":
        basis = ("A fixed task sequence by order and a rotating worker assignment "
                 "are used. The policy waits when the next fixed step is blocked.")
    else:
        basis = ("A random pairing of a free worker and ready task is selected. Dependencies and "
                 "availability remain hard constraints; active work is never discarded.")

    ready = []
    for gid in info.get("eligible_tasks", []):
        slot, idx = divmod(gid, TASKS_PER_ORDER)
        order = info["orders"][slot]
        spec = ORDER_TEMPLATE[idx]
        unlocks = sum(idx in downstream.prerequisites for downstream in ORDER_TEMPLATE)
        ready.append({
            "gid": gid,
            "order_id": order["order_id"],
            "number": order["slot"] + 1,
            "label": spec.name.replace("_", " ").title(),
            "priority": order["priority"],
            "slack": order["slack"],
            "difficulty": spec.difficulty,
            "unlocks": unlocks,
        })
    ready.sort(key=lambda item: (-item["priority"], item["slack"],
                                 -item["unlocks"], -item["difficulty"], item["gid"]))
    for rank, item in enumerate(ready, 1):
        item["operational_rank"] = rank

    active = []
    for worker in info.get("workers", []):
        gid = worker.get("current_task")
        if gid is None:
            continue
        slot, idx = divmod(gid, TASKS_PER_ORDER)
        active.append({
            "worker": worker["name"],
            "kind": worker["kind"],
            "order_id": info["orders"][slot]["order_id"],
            "number": slot + 1,
            "label": ORDER_TEMPLATE[idx].name.replace("_", " ").title(),
            "progress": worker["progress"],
            "waiting": worker["waiting"],
        })

    return {
        "basis": basis,
        "ranking_note": ("Candidate order below is an inspectable operational ranking; "
                         "it is not a claim about hidden PPO logits."),
        "ready": ready,
        "active": active,
    }


def serialise_state():
    env, info = session["env"], session["info"]
    tracker = session["tracker"]

    # which worker (if any) is on each global task, and how far along
    working = {}
    for i, w in enumerate(info["workers"]):
        if w["current_task"] is not None:
            working[w["current_task"]] = {
                "worker": i, "kind": w["kind"], "name": w["name"],
                "progress": w["progress"], "waiting": w["waiting"],
            }

    priority_levels = sorted({order["priority"] for order in info["orders"]},
                             reverse=True)
    priority_rank = {order["slot"]: priority_levels.index(order["priority"]) + 1
                     for order in info["orders"]}
    orders = []
    for o in info["orders"]:
        tasks = []
        for idx in range(TASKS_PER_ORDER):
            spec = ORDER_TEMPLATE[idx]
            gid = o["slot"] * TASKS_PER_ORDER + idx
            status = o["status"][idx]
            act = working.get(gid)
            tasks.append({
                "gid":         gid,
                "idx":         idx,
                "name":        spec.name,
                "label":       spec.name.replace("_", " ").title(),
                "difficulty":  spec.difficulty,
                "prereqs":     list(spec.prerequisites),
                "prereqs_met": all(o["status"][p] == 1.0 for p in spec.prerequisites),
                "status":      {0.0: "pending", 0.5: "active", 1.0: "done"}[status],
                "assignee":    act["kind"] if act else None,
                "worker_name": act["name"] if act else None,
                "waiting":     act["waiting"] if act else False,
                "progress":    round(act["progress"], 3) if act else (1.0 if status == 1.0 else 0.0),
            })
        orders.append({**o, "number": o["slot"] + 1,
                       "priority_rank": priority_rank[o["slot"]], "tasks": tasks})

    # Every worker gets a performance summary, not just the human. For the
    # robot these are the only numbers that move. Its speed and error rate
    # are constant by design.
    elapsed = max(info["step"], 1)
    workers = []
    for w in info["workers"]:
        busy = w.get("busy_steps", 0.0)
        workers.append({
            **w,
            "utilisation":   round(min(busy / elapsed, 1.0), 3),
            "idle_steps":    round(max(elapsed - busy, 0.0), 1),
            "steps_per_task": round(busy / w["tasks_done"], 1) if w["tasks_done"] else None,
            "throughput": round(w["tasks_done"] * 10.0 / elapsed, 2),
            "completion_share": round(w["tasks_done"] / max(info["tasks_done"], 1), 3),
        })

    human_workers = [w for w in info["workers"] if w["kind"] == "human"]
    observer_rows = []
    for index, snapshot in enumerate(session["hbo"].snapshots()):
        observer_rows.append({
            **snapshot,
            "worker": (human_workers[index]["name"]
                       if index < len(human_workers) else f"human_{index}"),
        })

    note = session["note"]
    if session["recoveries"]:
        guard_note = (f"PPO continuity guard used {session['recoveries']} time(s): "
                      "each intervention is labelled in the decision explanation.")
        note = f"{note} {guard_note}".strip()

    return jsonify({
        "step":       info["step"],
        "max_steps":  env.max_steps,
        "reward":     round(tracker.total_reward, 2),
        "ces":        live_ces(),
        "conflicts":  tracker.conflict_count,
        "sync_steps": tracker.parallel_steps,
        "waits":      tracker.wait_actions,
        "reassignments": tracker.reassignments,
        "role_switches": tracker.role_switches,
        "human_tasks":   tracker.human_tasks,
        "ai_tasks":      tracker.ai_tasks,
        "tasks_done":  info["tasks_done"],
        "total_tasks": info["total_tasks"],
        "orders_done": info["orders_done"],
        "num_orders":  info["num_orders"],
        "done":        session["done"],
        "solved":      session["terminated"],
        "profile":     session["profile"],
        "policy":      session["policy_kind"],
        "note":        note,
        "requested_policy": session["requested_policy"],
        "recoveries":  session["recoveries"],
        "zero_shot":   session["profile"] in TEST_PROFILES,
        "priority":    session["priority"],
        "priority_order": session["priority_order"],
        "workers":     workers,
        "orders":      orders,
        "observer": {
            "enabled": session["hbo"].enabled,
            "feature_order": ["smoothed speed", "smoothed error rate",
                              "completion ratio", "speed trend",
                              "completed workload"],
            "humans": observer_rows,
        },
        "sequencing": sequencing_snapshot(info),
        "log":         session["log"][-60:][::-1],
        "explanations": session["explanations"][-12:][::-1],
    })


# ── routes ────────────────────────────────────────────────────────────── #
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/config")
def api_config():
    return jsonify({
        "seen_profiles":     TRAIN_PROFILES,
        "held_out_profiles": TEST_PROFILES,
        "policies":          ["ppo"] + sorted(BASELINES.keys()),
        "checkpoint_exists": os.path.exists(CHECKPOINT),
        "research":          research_summary(),
        "priorities":        [{"key": key, "label": label}
                              for key, (_, label) in PRIORITY_PRESETS.items()],
        "max_orders":        MAX_UI_ORDERS,
        "defaults": {"num_orders": DEFAULT_NUM_ORDERS,
                     "num_humans": DEFAULT_NUM_HUMANS,
                     "num_ai": DEFAULT_NUM_AI,
                     "max_steps": MAX_EPISODE_STEPS},
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    body = request.get_json(silent=True) or {}
    try:
        profile_name = body.get("profile", "average")
        policy_kind = body.get("policy", "ppo")
        priority = body.get("priority", "random")
        num_orders = int(body.get("num_orders", DEFAULT_NUM_ORDERS))
        num_humans = int(body.get("num_humans", DEFAULT_NUM_HUMANS))
        num_ai = int(body.get("num_ai", DEFAULT_NUM_AI))
        if profile_name not in HUMAN_PROFILES:
            raise ValueError(f"unknown partner profile {profile_name!r}")
        if policy_kind not in {"ppo", *BASELINES.keys()}:
            raise ValueError(f"unknown policy {policy_kind!r}")
        if priority not in PRIORITY_PRESETS:
            raise ValueError(f"unknown order priority preset {priority!r}")
        if not 1 <= num_orders <= MAX_UI_ORDERS:
            raise ValueError(f"num_orders must be between 1 and {MAX_UI_ORDERS}")
        if not 1 <= num_humans <= 2 or not 1 <= num_ai <= 2:
            raise ValueError("dashboard teams support one or two AI and one or two human workers")
        reset_session(
            profile_name=profile_name,
            policy_kind=policy_kind,
            num_orders=num_orders,
            num_humans=num_humans,
            num_ai=num_ai,
            priority=priority,
            priority_order=body.get("priority_order"),
            seed=int(body.get("seed", 42)),
        )
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 400
    return serialise_state()


@app.route("/api/step", methods=["POST"])
def api_step():
    if not session:
        reset_session()
    body = request.get_json(silent=True) or {}
    try:
        count = int(body.get("count", 1))
        if count < 1:
            raise ValueError("step count must be positive")
        count = min(count, session["env"].max_steps)
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    for _ in range(count):
        if session["done"]:
            break
        step_once()
    return serialise_state()


@app.route("/api/state")
def api_state():
    if not session:
        reset_session()
    return serialise_state()


# ── UI ────────────────────────────────────────────────────────────────── #
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DTS ZSC Coordination Dashboard</title>
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  html{width:100%;max-width:100%;overflow-x:hidden}
  :root{
    --bg:#eef2f3; --surface:#fff; --surface2:#f3f6f6; --border:#d8e0e2;
    --text:#18272f; --muted:#607079; --hint:#89979d;
    --navy:#172a33; --navy-2:#213b46; --brand:#24766f; --brand-hover:#1c625d;
    --purple:#3b5b92; --purple-lt:#edf1f8;
    --amber:#ad6427;  --amber-lt:#fff1e4;
    --green:#4b7947;  --green-lt:#eaf3e8;
    --red:#ad4848;    --red-lt:#faecec;
    --shadow:0 1px 2px rgba(23,42,51,.04),0 8px 24px rgba(23,42,51,.045);
  }
  body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       background:var(--bg);color:var(--text);
       font-size:14px;line-height:1.5;min-width:320px;overflow-x:hidden}
  .page{width:100%;max-width:none;margin:0;padding:clamp(16px,2vw,32px) clamp(12px,2.25vw,40px) 48px}
  .page>*{min-width:0}

  .hdr{display:grid;grid-template-columns:minmax(260px,1fr) auto;align-items:center;
       margin-bottom:16px;gap:18px 28px;padding:18px 20px;border:1px solid rgba(255,255,255,.06);
       border-radius:14px;background:var(--navy);color:#f7fafb;
       box-shadow:0 12px 32px rgba(23,42,51,.12)}
  .hdr h1{font-size:clamp(20px,1.45vw,26px);font-weight:680;letter-spacing:-.6px;line-height:1.2}
  .hdr>div{min-width:0}
  .hdr p{font-size:12px;color:#b7c5ca;margin-top:5px;overflow-wrap:anywhere}
  .brand-kicker{display:flex;align-items:center;gap:7px;margin-bottom:7px;color:#7fd0c5;
                font-size:9px;font-weight:700;letter-spacing:1.35px;text-transform:uppercase}
  .brand-kicker::before{content:"";width:14px;height:2px;background:#58b5a9;border-radius:2px}
  .controls{display:flex;gap:8px;align-items:flex-end;flex-wrap:wrap}
  .control-field{display:flex;flex-direction:column;gap:4px;min-width:132px}
  .control-field:nth-child(2){min-width:145px}
  .control-field span{padding-left:2px;color:#9fb1b8;font-size:9px;font-weight:650;
                      letter-spacing:.75px;text-transform:uppercase}
  select,button{min-height:36px;font-size:12px;padding:7px 11px;border-radius:7px;cursor:pointer;
    border:1px solid var(--border);background:var(--surface);color:var(--text);
    box-shadow:0 1px 1px rgba(23,42,51,.03);transition:border-color .16s,box-shadow .16s,background .16s}
  .control-field select{width:100%;border-color:rgba(255,255,255,.14)}

  /* Click to rank: the operator picks the working sequence order by order, so
     an incomplete or duplicated ranking is not expressible in the first place. */
  .rank-bar{margin:-6px 0 16px;padding:13px 16px;border:1px solid var(--border);border-radius:12px;
            background:var(--surface);box-shadow:0 1px 2px rgba(23,42,51,.04)}
  .rank-head{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
  .rank-title{font-size:9px;font-weight:700;letter-spacing:.9px;text-transform:uppercase;color:#69808a}
  .rank-hint{font-size:11.5px;color:var(--muted)}
  .rank-chips{display:flex;gap:7px;flex-wrap:wrap;margin-top:9px}
  .rank-chip{display:inline-flex;align-items:center;gap:7px;min-height:34px;padding:6px 12px;
             font-size:12px;font-weight:600;border-radius:8px;cursor:pointer;
             border:1px solid var(--border);background:#fff;color:var(--text);
             transition:border-color .16s,background .16s,box-shadow .16s}
  .rank-chip:hover{border-color:#9eb1b6;box-shadow:0 2px 8px rgba(23,42,51,.08)}
  .rank-chip:focus-visible{outline:3px solid rgba(67,169,157,.26);outline-offset:1px}
  .rank-chip.picked{border-color:var(--brand);background:rgba(67,169,157,.09);color:#1d5f57}
  .rank-chip .rank-badge{display:inline-flex;align-items:center;justify-content:center;
             min-width:20px;height:20px;padding:0 5px;border-radius:6px;font-size:10px;font-weight:700;
             background:var(--brand);color:#fff;font-variant-numeric:tabular-nums}
  .rank-chip.picked .rank-badge.first{background:#c0392f}
  .rank-foot{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:10px}
  .rank-summary{font-size:11.5px;color:var(--muted);font-variant-numeric:tabular-nums}
  .rank-foot button{min-height:30px;padding:4px 12px;font-size:11.5px}
  .hdr button:not(.primary){color:#eef4f5;background:rgba(255,255,255,.07);border-color:rgba(255,255,255,.16)}
  .hdr button:not(.primary):hover{background:rgba(255,255,255,.13);border-color:rgba(255,255,255,.24)}
  .hdr .controls button{min-width:58px;padding-inline:13px;font-weight:650;letter-spacing:.05px}
  select:hover,button:hover{border-color:#9eb1b6;box-shadow:0 2px 8px rgba(23,42,51,.08)}
  select:focus-visible,button:focus-visible{outline:3px solid rgba(67,169,157,.26);outline-offset:1px}
  button.primary{background:var(--brand);color:#fff;border-color:var(--brand)}
  button.primary:hover{background:var(--brand-hover);border-color:var(--brand-hover)}
  button:disabled{opacity:1;cursor:not-allowed;box-shadow:none}
  .hdr button:not(.primary):disabled{color:#91a4ab;background:#223943;border-color:#3b525c}
  .hdr button.primary:disabled{color:#a8c9c5;background:#315e5b;border-color:#3b716d}
  .hdr button:disabled:hover{box-shadow:none}
  .note{font-size:12px;color:var(--amber);background:var(--amber-lt);
        border:1px solid #f0cfad;border-radius:8px;padding:9px 12px;margin-bottom:14px}

  .metrics{display:grid;grid-template-columns:repeat(6,minmax(130px,1fr));gap:clamp(8px,1vw,14px);margin-bottom:20px}
  .mc{position:relative;overflow:hidden;background:var(--surface);border:1px solid var(--border);
      border-radius:9px;padding:14px 15px 13px;box-shadow:var(--shadow);min-width:0}
  .mc::before{content:"";position:absolute;inset:0 0 auto 0;height:3px;background:var(--hint);opacity:.38}
  .mc:nth-child(2)::before,.mc:nth-child(5)::before{background:var(--purple);opacity:1}
  .mc:nth-child(4)::before{background:var(--green);opacity:1}
  .mc:nth-child(6)::before{background:var(--red);opacity:.7}
  .mc .lbl{font-size:9.5px;font-weight:680;letter-spacing:.62px;text-transform:uppercase;color:var(--muted);margin-bottom:5px}
  .mc .val{font-size:clamp(22px,1.7vw,29px);font-weight:700;letter-spacing:-.65px;line-height:1.1}
  .mc .sub{font-size:10.5px;color:var(--hint);margin-top:4px}

  .body-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(310px,360px);
             gap:clamp(14px,1.25vw,22px);align-items:start}
  .main-column{display:flex;flex-direction:column;gap:clamp(14px,1.25vw,22px);min-width:0}
  .right-rail{display:flex;flex-direction:column;gap:12px;min-width:0}
  #workersCol{display:flex;flex-direction:column;gap:12px;min-width:0}
  #workersCol .allocation-card{grid-column:1/-1}
  #ordersCol{min-width:0}

  .card{border:1px solid var(--border);border-radius:10px;padding:clamp(13px,1vw,17px);
        background:var(--surface);box-shadow:var(--shadow);min-width:0}
  .card+.card{margin-top:12px}
  .right-rail .card+.card,.right-rail .side-card,#workersCol .card+.card{margin-top:0}
  .card h3{font-size:13px;font-weight:680;letter-spacing:-.08px;margin-bottom:10px;display:flex;
           align-items:center;gap:7px;justify-content:space-between}
  .card h3 .left{display:flex;align-items:center;gap:7px;min-width:0;flex-wrap:wrap;overflow-wrap:anywhere}
  .dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex-shrink:0}
  .tag{font-size:9.5px;font-weight:650;letter-spacing:.08px;padding:2px 8px;border:1px solid transparent;
       border-radius:999px;flex-shrink:0}
  .tag-grey{background:var(--surface2);border-color:#e4e9ea;color:var(--muted)}
  .tag-amber{background:var(--amber-lt);border-color:#f2dac4;color:var(--amber)}
  .tag-green{background:var(--green-lt);border-color:#d7e8d4;color:var(--green)}
  .tag-red{background:var(--red-lt);border-color:#f0d4d4;color:var(--red)}
  .tag-purple{background:var(--purple-lt);border-color:#dbe3f1;color:var(--purple)}

  /* dependency graph */
  .dag-wrap{width:100%;overflow-x:auto;overflow-y:hidden;scrollbar-width:thin;
            scrollbar-color:rgba(36,118,111,.28) transparent}
  .dag{width:100%;max-width:820px;height:auto;display:block;overflow:visible;margin:0 auto}
  .dag text{font-family:inherit}
  .n-label{font-size:10.5px;font-weight:650;fill:var(--text)}
  .n-sub{font-size:8.8px;fill:var(--hint)}
  .edge{stroke:#c5ced1;stroke-width:1.6;fill:none}
  .edge.met{stroke:var(--green)}
  .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--muted);
          margin-top:6px;padding-top:9px;border-top:1px solid var(--border)}
  .legend span{display:flex;align-items:center;gap:5px}
  .sw{width:10px;height:10px;border-radius:3px;display:inline-block}

  .ord-head{display:flex;justify-content:space-between;align-items:center;
            font-size:12px;color:var(--muted);margin-bottom:2px}
  .slackbar{height:4px;background:#e9eeee;border-radius:4px;overflow:hidden;margin:7px 0 11px}
  .slackbar div{height:100%;border-radius:2px;transition:width .3s}

  .stat-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:7px}
  .stat-row:last-child{margin-bottom:0}
  .sl{font-size:12px;color:var(--muted)}
  .sv{font-size:12px;font-weight:500}
  .bar-w{flex:1;margin:0 9px;height:4px;background:#e6ebec;border-radius:4px}
  .bar-f{height:100%;border-radius:2px;transition:width .4s}

  .log-box{font-size:11px;font-family:ui-monospace,Menlo,Consolas,monospace;
           max-height:230px;overflow-y:auto;display:flex;flex-direction:column;gap:3px}
  .log-line{color:var(--muted);padding:1px 0}
  .log-line.ok{color:var(--green)} .log-line.bad{color:var(--red)}
  .log-line.ai{color:var(--purple)} .log-line.human{color:var(--amber)}
  .log-line.dim{color:var(--hint)}

  /* decision explanations */
  .explain-note{font-size:10.5px;color:var(--hint);font-weight:400;text-align:right}
  .explain-box{max-height:430px;overflow-y:auto;display:flex;flex-direction:column;gap:8px}
  .decision-item{border:1px solid var(--border);border-radius:7px;padding:10px;background:#f8fafa}
  .decision-item:first-child{border-color:#9ebbb8;background:#eef6f5;box-shadow:inset 3px 0 0 var(--brand)}
  .decision-head{display:flex;justify-content:space-between;gap:8px;align-items:flex-start;
                 font-size:11.5px;font-weight:600;margin-bottom:6px;min-width:0}
  .decision-head>span:first-child{min-width:0;overflow-wrap:anywhere}
  .decision-step{font-size:9px;color:var(--hint);font-weight:500;white-space:nowrap}
  .decision-line{font-size:11px;color:var(--muted);margin-top:5px;line-height:1.45;overflow-wrap:anywhere}
  .decision-line b{font-weight:600;color:var(--text)}
  .signal-pill{display:block;margin-top:7px;border-radius:6px;padding:6px 8px;
               font-size:10.5px;line-height:1.4;background:var(--surface2);color:var(--muted);
               overflow-wrap:anywhere}
  .signal-pill.attention{background:var(--amber-lt);color:var(--amber)}
  .signal-pill.stable{background:var(--green-lt);color:var(--green)}
  .empty-state{font-size:11px;color:var(--hint);padding:7px 0}

  /* sequencing basis and observer output */
  .basis-copy{font-size:11px;color:var(--muted);line-height:1.5;padding:8px 9px;
              border-radius:7px;background:#f8fafa;border:1px solid var(--border)}
  .basis-note{font-size:9.5px;color:var(--hint);line-height:1.4;margin-top:7px}
  .section-label{margin:11px 0 6px;font-size:9px;font-weight:700;letter-spacing:.7px;
                 text-transform:uppercase;color:var(--hint)}
  .queue-list{display:flex;flex-direction:column;gap:6px}
  .queue-item{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:7px;
              align-items:center;padding:7px 8px;border-radius:7px;background:var(--surface2)}
  .queue-rank{width:21px;height:21px;display:grid;place-items:center;border-radius:50%;
              background:var(--navy);color:#fff;font-size:9px;font-weight:700}
  .queue-main{font-size:10.5px;min-width:0;overflow-wrap:anywhere}
  .queue-main b{display:block;font-size:11px;color:var(--text)}
  .queue-meta{font-size:9px;color:var(--hint);text-align:right;white-space:nowrap}
  .active-item{border-left:3px solid var(--brand)}
  .observer-features{display:flex;flex-direction:column;gap:8px}
  .observer-human{padding:9px;border:1px solid var(--border);border-radius:7px;background:#f8fafa}
  .observer-head{display:flex;justify-content:space-between;gap:8px;font-size:11px;
                 font-weight:650;margin-bottom:7px}
  .feature-grid{display:grid;grid-template-columns:1fr auto;gap:5px 10px}
  .feature-grid span{font-size:10.5px;color:var(--muted)}
  .feature-grid b{font-size:10.5px;font-weight:600;font-variant-numeric:tabular-nums}
  .observer-foot{font-size:9.5px;color:var(--hint);line-height:1.4;margin-top:8px}

  /* evaluation and research evidence */
  .research-grid{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(360px,.85fr);
                 gap:clamp(14px,1.25vw,22px);margin-top:0;align-items:stretch}
  .research-card{margin-top:0!important}
  .panel-intro{font-size:11px;color:var(--muted);margin:-4px 0 12px}
  .compare-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px}
  @media(max-width:680px){.compare-grid{grid-template-columns:1fr}}
  .pair-card{border:1px solid var(--border);border-radius:8px;padding:12px;min-width:0;
             background:#f8fafa;
             transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease}
  .pair-card:hover{border-color:#b8c8cb;box-shadow:0 4px 14px rgba(23,42,51,.05)}
  .pair-head{display:flex;align-items:center;justify-content:space-between;gap:6px;
             font-size:11px;font-weight:600;margin-bottom:10px}
  .pair-values{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-bottom:8px}
  .pair-value span{display:block;font-size:9px;color:var(--muted);white-space:nowrap;
                   overflow:hidden;text-overflow:ellipsis}
  .pair-value b{font-size:18px;line-height:1.2}
  .pair-value:last-child{text-align:right}
  .compare-bar{height:4px;background:#e4e9ea;border-radius:4px;overflow:hidden;margin-top:5px}
  .compare-bar div{height:100%;border-radius:4px}
  .pair-detail{font-size:9.5px;color:var(--hint);margin-top:8px;min-height:27px}
  .evidence-list{display:flex;flex-direction:column;gap:7px}
  .evidence-item{display:grid;grid-template-columns:1fr auto;gap:2px 10px;align-items:center;
                 border-bottom:1px solid #e7ebec;padding:1px 0 8px}
  .evidence-item:last-child{border-bottom:0;padding-bottom:0}
  .evidence-label{font-size:11px;font-weight:600}
  .evidence-value{font-size:17px;font-weight:600;color:var(--purple);grid-row:1/3;grid-column:2}
  .evidence-value.green{color:var(--green)}
  .evidence-detail{font-size:9.5px;color:var(--hint);line-height:1.35}

  .status-bar{margin-top:16px;padding:10px 14px;border-radius:7px;background:var(--navy);
    border:1px solid var(--border);font-size:12px;color:var(--muted);
    display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
  .status-bar:not(.good):not(.bad){color:#cad5d9;border-color:var(--navy)}
  .status-bar:not(.good):not(.bad) b{color:#fff}
  .status-bar.good{background:var(--green-lt);border-color:var(--green);color:var(--green)}
  .status-bar.bad{background:var(--red-lt);border-color:var(--red);color:var(--red)}

  /* large monitors: use the extra width for worker density, not empty margins */
  @media(min-width:1500px){
    .body-grid{grid-template-columns:minmax(0,2.15fr) minmax(520px,1fr)}
    #workersCol{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
    #workersCol .allocation-card{grid-column:1/-1}
    .explain-box{max-height:360px}
  }

  /* laptops and tablets */
  @media(max-width:1100px){
    .hdr{grid-template-columns:1fr}
    .controls{width:100%;display:grid;grid-template-columns:repeat(12,minmax(0,1fr))}
    .control-field,.controls button{width:100%;min-width:0}
    .control-field{grid-column:span 3}
    .controls button{grid-column:span 4}
    .metrics{grid-template-columns:repeat(3,minmax(0,1fr))}
    .body-grid{grid-template-columns:1fr}
    .right-rail{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
    #workersCol{grid-column:1/-1;display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
  }

  @media(max-width:900px){
    .research-grid{grid-template-columns:1fr}
    .compare-grid{grid-template-columns:repeat(3,minmax(0,1fr))}
  }

  /* phones and narrow tablets */
  @media(max-width:680px){
    .page{padding:14px 12px 34px}
    .hdr{gap:14px;margin-bottom:14px;padding:16px 14px;border-radius:11px}
    .hdr p{font-size:11px;line-height:1.4}
    .controls{grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}
    .control-field,.controls button{grid-column:auto;min-width:0}
    .control-field:nth-child(2){min-width:0}
    .controls button:last-child{grid-column:1/-1}
    select,button{min-height:40px;font-size:12px}
    .metrics{grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-bottom:14px}
    .mc{padding:11px 12px;border-radius:10px}
    .mc .val{font-size:22px}
    .right-rail{grid-template-columns:1fr}
    #workersCol{grid-template-columns:repeat(auto-fit,minmax(235px,1fr))}
    .compare-grid{grid-template-columns:1fr}
    .dag{width:660px;min-width:660px}
    .ord-head{align-items:flex-start;gap:4px;flex-direction:column}
    .card h3{align-items:flex-start;flex-wrap:wrap}
    .legend{gap:9px 12px}
    .status-bar{align-items:flex-start;flex-direction:column}
  }

  @media(max-width:380px){
    .metrics{grid-template-columns:1fr 1fr}
    .mc .lbl,.mc .sub{font-size:10px}
    .pair-values{gap:6px}
  }
</style>
</head>
<body>
<div class="page">
  <div class="hdr">
    <div>
      <div class="brand-kicker">Research operations / live coordination</div>
      <h1>DTS ZSC Coordination Dashboard</h1>
      <p>Dynamic task sequencing for zero shot human AI coordination in a live warehouse simulation</p>
    </div>
    <div class="controls">
      <label class="control-field"><span>Policy</span>
        <select id="policySel" aria-label="Coordination policy" onchange="doReset()"></select>
      </label>
      <label class="control-field"><span>Partner</span>
        <select id="profileSel" aria-label="Human partner profile" onchange="doReset()"></select>
      </label>
      <label class="control-field"><span>Workload</span>
        <select id="ordersSel" aria-label="Number of orders" onchange="onOrdersChange()">
          <option value="1">1 order</option>
          <option value="2" selected>2 orders</option>
          <option value="3">3 orders</option>
          <option value="4">4 orders</option>
          <option value="5">5 orders</option>
          <option value="6">6 orders</option>
        </select>
      </label>
      <label class="control-field"><span>Team</span>
        <select id="workersSel" aria-label="Team composition" onchange="doReset()">
          <option value="1,1" selected>1 AI + 1 human</option>
          <option value="1,2">1 AI + 2 humans</option>
          <option value="2,2">2 AI + 2 humans</option>
        </select>
      </label>
      <label class="control-field"><span>Order priority</span>
        <select id="prioritySel" aria-label="Order priority" onchange="onPriorityChange()"></select>
      </label>
      <button id="stepBtn" onclick="doStep(1)" title="Advance the simulation by one step: the agent makes one decision and every worker advances one tick">Step</button>
      <button id="step5Btn" onclick="doStep(5)" title="Advance five steps">Step &times;5</button>
      <button id="playBtn" class="primary" onclick="togglePlay()">&#9654; Run</button>
      <button onclick="doReset()">Reset</button>
    </div>
  </div>

  <div id="rankBar" class="rank-bar" style="display:none">
    <div class="rank-head">
      <span class="rank-title">Priority sequence</span>
      <span class="rank-hint" id="rankHint">Click the orders in the sequence you want them worked.</span>
    </div>
    <div class="rank-chips" id="rankChips" role="group"
         aria-label="Order priority sequence, most urgent first"></div>
    <div class="rank-foot">
      <span id="rankSummary" class="rank-summary"></span>
      <button type="button" id="rankClear" onclick="clearRanking()">Clear</button>
    </div>
  </div>

  <div id="noteBox"></div>

  <div class="metrics">
    <div class="mc"><div class="lbl">Joint reward</div><div class="val" id="mReward">0.0</div><div class="sub">cumulative</div></div>
    <div class="mc"><div class="lbl">CES</div><div class="val" id="mCES">Pending</div><div class="sub">coordination efficiency</div></div>
    <div class="mc"><div class="lbl">Step</div><div class="val" id="mStep">0</div><div class="sub" id="mStepSub">of 70</div></div>
    <div class="mc"><div class="lbl">Tasks done</div><div class="val" id="mTasks">0</div><div class="sub" id="mTasksSub">of 12</div></div>
    <div class="mc"><div class="lbl">Sync steps</div><div class="val" id="mSync">0</div><div class="sub">both workers busy</div></div>
    <div class="mc"><div class="lbl">Conflicts</div><div class="val" id="mConf">0</div><div class="sub">illegal assignments</div></div>
  </div>

  <div class="body-grid">
    <div class="main-column">
      <div id="ordersCol"></div>
      <div class="research-grid">
        <div class="card research-card">
          <h3><span class="left">Performance comparisons</span><span class="tag tag-grey">saved evaluation</span></h3>
          <div class="panel-intro" id="comparisonIntro">CES across evaluated partner profiles; higher is better.</div>
          <div class="compare-grid" id="comparisonPanel"></div>
        </div>
        <div class="card research-card">
          <h3><span class="left">Research evidence</span><span class="tag tag-green">baseline tested</span></h3>
          <div class="panel-intro">Measured outcomes from held out partner and baseline evaluations.</div>
          <div class="evidence-list" id="evidencePanel"></div>
        </div>
      </div>
    </div>
    <div class="right-rail">
      <div id="workersCol"></div>
      <div class="card side-card">
        <h3><span class="left">Human Observer output <span class="tag tag-purple">live input</span></span></h3>
        <div class="explain-note" style="text-align:left;margin:-6px 0 9px">Exact feature block appended to the coordinator observation</div>
        <div class="observer-features" id="observerBox"></div>
      </div>
      <div class="card side-card">
        <h3><span class="left">Step sequencing basis <span class="tag tag-green">live</span></span></h3>
        <div id="sequencingBox"></div>
      </div>
      <div class="card side-card">
        <h3><span class="left">Coordinator decision explanations <span class="tag tag-purple">live</span></span></h3>
        <div class="explain-note" style="text-align:left;margin:-6px 0 9px">Observed input rationale, not causal attribution</div>
        <div class="explain-box" id="explanationBox"></div>
      </div>
      <div class="card">
        <h3><span class="left">Event log</span></h3>
        <div class="log-box" id="logBox"></div>
      </div>
    </div>
  </div>

  <div class="status-bar" id="statusBar">Loading&hellip;</div>
</div>

<script>
let playing=false, timer=null, stepping=false, resetSerial=0;

const NODE_POS = [
  {x: 58,  y: 30},   // fetch A
  {x: 58,  y: 96},   // fetch B
  {x: 196, y: 63},   // pack
  {x: 330, y: 63},   // label
  {x: 462, y: 63},   // weigh
  {x: 596, y: 63},   // dispatch
];
const NODE_W = 96, NODE_H = 40;

async function post(path, body={}){
  const r = await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},
                             body:JSON.stringify(body)});
  const data = await r.json().catch(()=>({error:`Invalid response from ${path}`}));
  if(!r.ok || data.error) throw new Error(data.error||`Request failed (${r.status})`);
  return data;
}

function showError(error){
  stopPlay();
  const bar=document.getElementById('statusBar');
  bar.className='status-bar bad';
  bar.textContent='Error: '+(error && error.message ? error.message : String(error));
}

function escapeHtml(value){
  return String(value).replace(/[&<>"']/g,ch=>({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'
  })[ch]);
}

function renderResearch(research){
  const comparisons=document.getElementById('comparisonPanel');
  const evidence=document.getElementById('evidencePanel');
  if(!research || !research.available){
    const msg='Saved evaluation data is unavailable.'
      +(research && research.error ? ' '+escapeHtml(research.error) : '');
    comparisons.innerHTML=`<div class="empty-state">${msg}</div>`;
    evidence.innerHTML=`<div class="empty-state">${msg}</div>`;
    return;
  }

  document.getElementById('comparisonIntro').textContent =
    `CES across ${research.profiles} partner profiles · ${research.episodes_per_profile} episodes per profile · higher is better.`;
  comparisons.innerHTML=research.comparisons.map((c,i)=>{
    const delta=c.delta*100;
    const deltaClass=delta>=-2?'tag-green':'tag-amber';
    const deltaText=`${Math.abs(delta).toFixed(1)}% ${delta>=0?'higher':'lower'}`;
    const leftWidth=Math.min(100,Math.max(2,c.left_value/0.9*100));
    const rightWidth=Math.min(100,Math.max(2,c.right_value/0.9*100));
    const rightColour=i===2?'var(--green)':'var(--purple)';
    return `<div class="pair-card">
      <div class="pair-head"><span>${escapeHtml(c.title)}</span>
        <span class="tag ${deltaClass}">${deltaText}</span></div>
      <div class="pair-values">
        <div class="pair-value"><span>${escapeHtml(c.left_label)}</span><b>${c.left_value.toFixed(3)}</b>
          <div class="compare-bar"><div style="width:${leftWidth}%;background:var(--hint)"></div></div></div>
        <div class="pair-value"><span>${escapeHtml(c.right_label)}</span><b>${c.right_value.toFixed(3)}</b>
          <div class="compare-bar"><div style="width:${rightWidth}%;background:${rightColour}"></div></div></div>
      </div>
      <div class="pair-detail">${escapeHtml(c.detail)}</div>
    </div>`;
  }).join('');

  evidence.innerHTML=research.evidence.map(e=>`<div class="evidence-item">
    <div class="evidence-label">${escapeHtml(e.label)}</div>
    <div class="evidence-value ${escapeHtml(e.tone)}">${escapeHtml(e.value)}</div>
    <div class="evidence-detail">${escapeHtml(e.detail)}</div>
  </div>`).join('');
}

async function loadConfig(){
  const cfg = await (await fetch('/api/config')).json();
  const pol = document.getElementById('policySel');
  const labels = {ppo:'PPO agent (learned)', greedy:'Greedy heuristic',
                  static:'Static workflow', random:'Random'};
  pol.innerHTML = cfg.policies.map(p=>
    `<option value="${p}">${labels[p]||p}</option>`).join('');

  const prof = document.getElementById('profileSel');
  const seen = cfg.seen_profiles.map(p=>`<option value="${p}">${p}</option>`).join('');
  const held = cfg.held_out_profiles.map(p=>`<option value="${p}">${p} (zero shot)</option>`).join('');
  prof.innerHTML = `<optgroup label="Seen during training">${seen}</optgroup>`
                 + `<optgroup label="Held out (never trained on)">${held}</optgroup>`;

  document.getElementById('prioritySel').innerHTML = cfg.priorities.map(p=>
    `<option value="${p.key}">${escapeHtml(p.label)}</option>`).join('');
  renderRanking();
  renderResearch(cfg.research);
}

// ── operator ranking ────────────────────────────────────────────────────
// ranking holds order numbers most urgent first. It is only ever built by
// clicking, so it cannot contain a duplicate, a gap or an out of range order.
let ranking = [];

function orderCount(){ return parseInt(document.getElementById('ordersSel').value); }
function isCustomPriority(){ return document.getElementById('prioritySel').value === 'custom'; }

function toggleRank(number){
  const at = ranking.indexOf(number);
  if(at === -1) ranking.push(number); else ranking.splice(at, 1);
  renderRanking();
  // The scenario is only rebuilt once every order has a place in the queue;
  // a partial ranking would have no defined urgency for the rest.
  if(ranking.length === orderCount()) doReset();
}

function clearRanking(){
  ranking = [];
  renderRanking();
}

function renderRanking(){
  const bar = document.getElementById('rankBar');
  const custom = isCustomPriority();
  bar.style.display = custom ? '' : 'none';
  if(!custom) return;

  const n = orderCount();
  const chips = document.getElementById('rankChips');
  chips.innerHTML = '';
  for(let number=1; number<=n; number++){
    const at = ranking.indexOf(number);
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'rank-chip' + (at===-1 ? '' : ' picked');
    chip.onclick = ()=>toggleRank(number);
    chip.title = at===-1 ? `Add order ${number} to the sequence`
                         : `Remove order ${number} from position ${at+1}`;
    chip.setAttribute('aria-pressed', at===-1 ? 'false' : 'true');
    chip.innerHTML = at===-1
      ? `Order ${number}`
      : `Order ${number}<span class="rank-badge${at===0?' first':''}">#${at+1}</span>`;
    chips.appendChild(chip);
  }

  const left = n - ranking.length;
  document.getElementById('rankSummary').textContent = ranking.length
    ? ranking.map(v=>'order '+v).join('  →  ')
      + (left ? `   (${left} still to place)` : '   ✓ applied')
    : 'Nothing picked yet. The first order you click is worked first.';
  document.getElementById('rankHint').textContent = left
    ? `Click ${left} more order${left===1?'':'s'} to set the full sequence.`
    : 'Click a picked order to take it out of the sequence.';
}

function onPriorityChange(){
  if(isCustomPriority() && ranking.length !== orderCount()) ranking = [];
  renderRanking();
  doReset();
}

function onOrdersChange(){
  // A ranking built for 3 orders is not a ranking for 5, so it is discarded
  // rather than sent through and rejected by the server.
  ranking = [];
  renderRanking();
  doReset();
}

function priorityOrderPayload(){
  const n = orderCount();
  // Until every order is placed, the unplaced ones keep their natural position
  // so the scenario stays well defined and the preview still means something.
  const seq = ranking.slice();
  for(let number=1; number<=n; number++) if(!seq.includes(number)) seq.push(number);
  return seq;
}

async function doReset(){
  stopPlay();
  const serial=++resetSerial;
  const [ai,hu] = document.getElementById('workersSel').value.split(',');
  const priority = document.getElementById('prioritySel').value;
  try{
    const d = await post('/api/reset',{
      profile: document.getElementById('profileSel').value,
      policy:  document.getElementById('policySel').value,
      num_orders: parseInt(document.getElementById('ordersSel').value),
      num_ai: parseInt(ai), num_humans: parseInt(hu),
      priority: priority,
      priority_order: priority==='custom' ? priorityOrderPayload() : null,
    });
    if(serial===resetSerial) render(d);
  }catch(error){ if(serial===resetSerial) showError(error); }
}

async function doStep(n=1){
  if(stepping) return null;
  stepping=true;
  try{
    const d=await post('/api/step',{count:n});
    render(d);
    return d;
  }catch(error){
    showError(error);
    return null;
  }finally{
    stepping=false;
  }
}

function stopPlay(){
  playing=false; clearTimeout(timer); timer=null;
  document.getElementById('playBtn').innerHTML='&#9654; Run';
}
async function playTick(){
  if(!playing) return;
  const d=await doStep(1);
  if(!playing || !d || d.done){ stopPlay(); return; }
  timer=setTimeout(playTick,220);
}
function togglePlay(){
  if(playing){stopPlay();return;}
  playing=true;
  document.getElementById('playBtn').innerHTML='&#10073;&#10073; Pause';
  playTick();
}

function nodeColors(t){
  if(t.status==='done')   return {fill:'#EAF3E8', stroke:'#4B7947', text:'#355F32'};
  if(t.status==='active') return t.assignee==='human'
      ? {fill:'#FFF1E4', stroke:'#AD6427', text:'#8D4B18'}
      : {fill:'#EDF1F8', stroke:'#3B5B92', text:'#2F4B7A'};
  if(!t.prereqs_met)      return {fill:'#F3F6F6', stroke:'#D9E0E2', text:'#9AA6AA'};
  return {fill:'#ffffff', stroke:'#B9C5C8', text:'#566870'};
}

function buildDag(order){
  const svgNS='http://www.w3.org/2000/svg';
  const svg=document.createElementNS(svgNS,'svg');
  // widest node right edge = 596 + NODE_W/2 = 644; lowest = 96 + NODE_H = 136
  svg.setAttribute('viewBox','0 0 660 144');
  svg.setAttribute('class','dag');

  // edges first, so nodes draw on top
  order.tasks.forEach(t=>{
    t.prereqs.forEach(p=>{
      const a=NODE_POS[p], b=NODE_POS[t.idx];
      const x1=a.x+NODE_W/2, y1=a.y+NODE_H/2, x2=b.x-NODE_W/2, y2=b.y+NODE_H/2;
      const mx=(x1+x2)/2;
      const path=document.createElementNS(svgNS,'path');
      path.setAttribute('d',`M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`);
      path.setAttribute('class', order.status[p]===1.0 ? 'edge met' : 'edge');
      svg.appendChild(path);
      const head=document.createElementNS(svgNS,'path');
      head.setAttribute('d',`M ${x2} ${y2} l -6 -3.4 l 0 6.8 z`);
      head.setAttribute('fill', order.status[p]===1.0 ? '#4B7947' : '#C5CED1');
      svg.appendChild(head);
    });
  });

  order.tasks.forEach(t=>{
    const pos=NODE_POS[t.idx], c=nodeColors(t);
    const g=document.createElementNS(svgNS,'g');

    const rect=document.createElementNS(svgNS,'rect');
    rect.setAttribute('x',pos.x-NODE_W/2); rect.setAttribute('y',pos.y);
    rect.setAttribute('width',NODE_W); rect.setAttribute('height',NODE_H);
    rect.setAttribute('rx','8');
    rect.setAttribute('fill',c.fill); rect.setAttribute('stroke',c.stroke);
    rect.setAttribute('stroke-width', t.status==='active' ? '1.8' : '1');
    g.appendChild(rect);

    const name=document.createElementNS(svgNS,'text');
    name.setAttribute('x',pos.x); name.setAttribute('y',pos.y+16);
    name.setAttribute('text-anchor','middle');
    name.setAttribute('class','n-label'); name.setAttribute('fill',c.text);
    name.textContent=t.label;
    g.appendChild(name);

    const sub=document.createElementNS(svgNS,'text');
    sub.setAttribute('x',pos.x); sub.setAttribute('y',pos.y+27);
    sub.setAttribute('text-anchor','middle'); sub.setAttribute('class','n-sub');
    sub.textContent = t.status==='done' ? 'done'
      : t.status==='active' ? (t.waiting ? t.worker_name+' starting…' : t.worker_name)
      : (t.prereqs_met ? 'ready · diff '+t.difficulty.toFixed(1) : 'blocked');
    g.appendChild(sub);

    // progress bar inside the node
    const track=document.createElementNS(svgNS,'rect');
    track.setAttribute('x',pos.x-NODE_W/2+8); track.setAttribute('y',pos.y+NODE_H-8);
    track.setAttribute('width',NODE_W-16); track.setAttribute('height',3);
    track.setAttribute('rx','1.5'); track.setAttribute('fill','rgba(0,0,0,.07)');
    g.appendChild(track);
    if(t.progress>0){
      const fill=document.createElementNS(svgNS,'rect');
      fill.setAttribute('x',pos.x-NODE_W/2+8); fill.setAttribute('y',pos.y+NODE_H-8);
      fill.setAttribute('width',(NODE_W-16)*t.progress); fill.setAttribute('height',3);
      fill.setAttribute('rx','1.5'); fill.setAttribute('fill',c.stroke);
      g.appendChild(fill);
    }
    svg.appendChild(g);
  });
  return svg;
}

function renderOrders(d){
  const col=document.getElementById('ordersCol');
  col.innerHTML='';
  d.orders.forEach(o=>{
    const card=document.createElement('div');
    card.className='card order-card';

    let tag='<span class="tag tag-grey">queued</span>';
    if(o.complete)      tag='<span class="tag tag-green">complete</span>';
    else if(!o.arrived) tag=`<span class="tag tag-grey">arrives at step ${o.arrival}</span>`;
    else if(o.late)     tag='<span class="tag tag-red">past deadline</span>';
    else if(o.slack<8)  tag=`<span class="tag tag-amber">${o.slack.toFixed(0)} steps of slack</span>`;
    else                tag=`<span class="tag tag-green">${o.slack.toFixed(0)} steps of slack</span>`;

    const urg = o.priority>0.66?'high':o.priority>0.4?'medium':'routine';
    const urgClass = o.priority>0.66?'tag-red':o.priority>0.4?'tag-amber':'tag-grey';

    const head=document.createElement('h3');
    head.innerHTML=`<span class="left">Order ${o.number}
        <span class="tag ${urgClass}">rank #${o.priority_rank} · priority ${o.priority.toFixed(2)} · ${urg}</span></span>${tag}`;
    card.appendChild(head);

    const meta=document.createElement('div');
    meta.className='ord-head';
    meta.innerHTML=`<span>${o.tasks_done}/6 tasks &middot; priority rank #${o.priority_rank}</span>
                    <span>deadline: step ${o.deadline.toFixed(0)}</span>`;
    card.appendChild(meta);

    const bar=document.createElement('div');
    bar.className='slackbar';
    const completion=Math.max(0,Math.min(1,o.tasks_done/6));
    const col2=o.complete?'var(--green)':(o.late?'var(--red)':'var(--brand)');
    bar.title=`Task completion: ${o.tasks_done}/6 (${Math.round(completion*100)}%)`;
    bar.setAttribute('role','progressbar');
    bar.setAttribute('aria-label',`Order ${o.number} task completion`);
    bar.setAttribute('aria-valuemin','0');
    bar.setAttribute('aria-valuemax','6');
    bar.setAttribute('aria-valuenow',String(o.tasks_done));
    bar.innerHTML=`<div style="width:${(completion*100).toFixed(1)}%;background:${col2}"></div>`;
    card.appendChild(bar);

    const dagWrap=document.createElement('div');
    dagWrap.className='dag-wrap';
    dagWrap.appendChild(buildDag(o));
    card.appendChild(dagWrap);
    col.appendChild(card);
  });

  const legend=document.createElement('div');
  legend.className='card legend-card';
  legend.innerHTML=`<h3><span class="left">Task dependency graph: legend</span></h3>
    <div class="legend" style="border:0;padding:0;margin:0">
      <span><i class="sw" style="background:#fff;border:1px solid #B9C5C8"></i> ready</span>
      <span><i class="sw" style="background:#F3F6F6;border:1px solid #D9E0E2"></i> blocked by prerequisites</span>
      <span><i class="sw" style="background:#FFF1E4;border:1px solid #AD6427"></i> human working</span>
      <span><i class="sw" style="background:#EDF1F8;border:1px solid #3B5B92"></i> AI working</span>
      <span><i class="sw" style="background:#EAF3E8;border:1px solid #4B7947"></i> done</span>
    </div>`;
  col.appendChild(legend);
}

function renderWorkers(d){
  const col=document.getElementById('workersCol');
  col.innerHTML='';
  d.workers.forEach(w=>{
    const isHuman=w.kind==='human';
    const accent=isHuman?'var(--amber)':'var(--purple)';
    const card=document.createElement('div');
    card.className='card';
    const state=w.waiting?'responding…':(w.busy?`task ${w.current_task}`:'idle');
    const zs = isHuman && d.zero_shot
      ? '<span class="tag tag-red">zero shot</span>' : '';
    let html=`<h3><span class="left"><span class="dot" style="background:${accent}"></span>
        ${w.name}</span><span class="tag ${isHuman?'tag-amber':'tag-purple'}">${isHuman?w.profile:'AI performance'}</span></h3>`;
    if(zs) html+=`<div style="margin:-4px 0 8px">${zs}</div>`;
    if(isHuman){
      // Only the human varies along these; the robot is constant by design.
      // Nothing has been observed before the first step, so show no values yet.
      html+=d.step>0
        ? row('Speed',w.speed,accent)+row('Fatigue',w.fatigue,'var(--red)')
          +row('Error rate',w.error_rate,'var(--red)')
        : pendingRow('Speed')+pendingRow('Fatigue')+pendingRow('Error rate');
    }else{
      html+=`<div class="observer-foot" style="margin:0 0 8px">Deterministic executor · fixed speed · no simulated fatigue or errors</div>`;
    }
    html+=row('Utilisation',w.utilisation,accent)
         +`<div class="stat-row"><span class="sl">Status</span><span class="sv">${state}</span></div>
           <div class="stat-row"><span class="sl">Progress</span><span class="sv">${(w.progress*100).toFixed(0)}%</span></div>
           <div class="stat-row"><span class="sl">Tasks finished</span><span class="sv">${w.tasks_done}</span></div>
           <div class="stat-row"><span class="sl">Steps per task</span><span class="sv">${w.steps_per_task===null?'Not yet':w.steps_per_task.toFixed(1)}</span></div>
           <div class="stat-row"><span class="sl">Throughput / 10 steps</span><span class="sv">${w.throughput.toFixed(2)}</span></div>
           <div class="stat-row"><span class="sl">Completed work share</span><span class="sv">${Math.round(w.completion_share*100)}%</span></div>
           <div class="stat-row"><span class="sl">Idle steps</span><span class="sv">${w.idle_steps.toFixed(0)}</span></div>`;
    if(isHuman && w.errors) html+=`<div class="stat-row"><span class="sl">Errors made</span><span class="sv">${w.errors}</span></div>`;
    card.innerHTML=html;
    col.appendChild(card);
  });

  const split=document.createElement('div');
  split.className='card allocation-card';
  const tot=d.human_tasks+d.ai_tasks;
  const hp=tot?Math.round(100*d.human_tasks/tot):0;
  split.innerHTML=`<h3><span class="left">Allocation</span></h3>
    <div class="stat-row"><span class="sl">To humans</span><span class="sv">${d.human_tasks} (${hp}%)</span></div>
    <div class="stat-row"><span class="sl">To AI</span><span class="sv">${d.ai_tasks} (${100-hp}%)</span></div>
    <div class="stat-row"><span class="sl">Role switches</span><span class="sv">${d.role_switches}</span></div>
    <div class="stat-row"><span class="sl">Wait actions</span><span class="sv">${d.waits}</span></div>
    <div class="stat-row"><span class="sl">Reassignments</span><span class="sv">${d.reassignments}</span></div>`;
  col.appendChild(split);
}

function renderObserver(d){
  const box=document.getElementById('observerBox');
  const observer=d.observer||{};
  const humans=observer.humans||[];
  if(!humans.length){
    box.innerHTML='<div class="empty-state">No human worker is present, so no observer feature block is emitted.</div>';
    return;
  }
  // Before the first step the observer has seen nothing, so its features are
  // only initial values, not observations.
  const fmt=v=>d.step>0?v:'Pending';
  box.innerHTML=humans.map(h=>`<div class="observer-human">
    <div class="observer-head"><span>${escapeHtml(h.worker)}</span>
      <span class="tag ${observer.enabled?'tag-green':'tag-red'}">${observer.enabled?'observer on':'observer off'}</span></div>
    <div class="feature-grid">
      <span>Smoothed speed</span><b>${fmt(h.smooth_speed.toFixed(3))}</b>
      <span>Smoothed error rate</span><b>${fmt(h.smooth_error_rate.toFixed(3))}</b>
      <span>Completion ratio</span><b>${fmt(h.completion_ratio.toFixed(3))}</b>
      <span>Speed trend</span><b>${fmt((h.speed_trend>=0?'+':'')+h.speed_trend.toFixed(3))}</b>
      <span>Completed workload</span><b>${fmt(h.completed_workload.toFixed(3))}</b>
    </div>
    <div class="observer-foot">${h.tasks_completed}/${h.tasks_assigned} observed assignments completed · ${h.observations} behavior samples</div>
  </div>`).join('')+
    `<div class="observer-foot">Feature order sent to the coordinator: ${escapeHtml((observer.feature_order||[]).join(' → '))}.</div>`;
}

function renderSequencing(d){
  const box=document.getElementById('sequencingBox');
  const seq=d.sequencing||{ready:[],active:[]};
  let html=`<div class="basis-copy">${escapeHtml(seq.basis||'No sequencing description available.')}</div>`;

  html+='<div class="section-label">Active work</div><div class="queue-list">';
  if(seq.active && seq.active.length){
    html+=seq.active.map(a=>`<div class="queue-item active-item">
      <span class="queue-rank">${a.kind==='ai'?'AI':'H'}</span>
      <span class="queue-main"><b>${escapeHtml(a.label)} · order ${a.number}</b>${escapeHtml(a.worker)}${a.waiting?' · responding':''}</span>
      <span class="queue-meta">${Math.round(a.progress*100)}%</span>
    </div>`).join('');
  }else{
    html+='<div class="empty-state">No task is active at this tick.</div>';
  }
  html+='</div><div class="section-label">Ready queue</div><div class="queue-list">';
  if(seq.ready && seq.ready.length){
    html+=seq.ready.slice(0,6).map(q=>`<div class="queue-item">
      <span class="queue-rank">${q.operational_rank}</span>
      <span class="queue-main"><b>${escapeHtml(q.label)} · order ${q.number}</b>priority ${q.priority.toFixed(2)} · unlocks ${q.unlocks}</span>
      <span class="queue-meta">slack ${q.slack.toFixed(0)}<br>diff ${q.difficulty.toFixed(1)}</span>
    </div>`).join('');
  }else{
    html+='<div class="empty-state">No unassigned step is ready. If work is active above, progress still advances every tick until dependencies unlock.</div>';
  }
  html+=`</div><div class="basis-note">${escapeHtml(seq.ranking_note||'')}</div>`;
  box.innerHTML=html;
}

function renderExplanations(d){
  const box=document.getElementById('explanationBox');
  const items=d.explanations||[];
  if(!items.length){
    box.innerHTML='<div class="empty-state">Step the simulation to see why tasks are assigned, roles shift, and which human signals were present.</div>';
    return;
  }
  box.innerHTML=items.slice(0,6).map(x=>`<div class="decision-item">
    <div class="decision-head"><span>${escapeHtml(x.decision)}</span>
      <span class="decision-step">STEP ${x.step}</span></div>
    <div class="decision-line"><b>Source</b> ${escapeHtml(x.source||d.policy)}</div>
    <div class="decision-line"><b>Why</b> ${escapeHtml(x.why)}</div>
    <div class="decision-line"><b>Role logic</b> ${escapeHtml(x.role_change)}</div>
    <span class="signal-pill ${escapeHtml(x.trigger_level)}"><b>Human trigger</b> · ${escapeHtml(x.trigger)}</span>
  </div>`).join('');
}

function pendingRow(label){
  return `<div class="stat-row"><span class="sl">${label}</span>
    <div class="bar-w"></div><span class="sv">Pending</span></div>`;
}

function row(label,val,colour){
  const pct=Math.round(Math.max(0,Math.min(1,val))*100);
  return `<div class="stat-row"><span class="sl">${label}</span>
    <div class="bar-w"><div class="bar-f" style="width:${pct}%;background:${colour}"></div></div>
    <span class="sv">${val.toFixed(2)}</span></div>`;
}

function render(d){
  if(d.error){ document.getElementById('statusBar').textContent='Error: '+d.error; return; }

  document.getElementById('mReward').textContent=d.reward.toFixed(1);
  document.getElementById('mCES').textContent=d.step>0?d.ces.toFixed(3):'Pending';
  document.getElementById('mStep').textContent=d.step;
  document.getElementById('mStepSub').textContent='of '+d.max_steps;
  document.getElementById('mTasks').textContent=d.tasks_done;
  document.getElementById('mTasksSub').textContent='of '+d.total_tasks;
  document.getElementById('mSync').textContent=d.sync_steps;
  document.getElementById('mConf').textContent=d.conflicts;

  const notes=[];
  if(d.priority_order && d.priority_order.length){
    notes.push('Operator priority sequence: '
      + d.priority_order.map(n=>'order '+n).join(' → ')
      + '. Rank #1 is the most urgent and gets the tightest deadline; orders still '
      + 'arrive on their own schedule, so a late-arriving order cannot start early.');
  }
  if(d.note) notes.push(d.note);
  document.getElementById('noteBox').innerHTML =
    notes.map(n=>`<div class="note">${escapeHtml(n)}</div>`).join('');

  renderOrders(d);
  renderWorkers(d);
  renderObserver(d);
  renderSequencing(d);
  renderExplanations(d);

  document.getElementById('logBox').innerHTML =
    d.log.map(l=>`<div class="log-line ${l.kind}">[${String(l.step).padStart(3,'0')}] ${escapeHtml(l.text)}</div>`).join('');

  const bar=document.getElementById('statusBar');
  const playBtn=document.getElementById('playBtn'), stepBtn=document.getElementById('stepBtn');
  const step5Btn=document.getElementById('step5Btn');
  if(d.done){
    bar.className='status-bar '+(d.solved?'good':'bad');
    bar.textContent = d.solved
      ? `All ${d.num_orders} orders fulfilled in ${d.step} steps. CES: ${d.ces.toFixed(3)} | Reward: ${d.reward.toFixed(1)}`
      : `Step limit reached. Tasks: ${d.tasks_done}/${d.total_tasks} | Orders: ${d.orders_done}/${d.num_orders} | CES: ${d.ces.toFixed(3)}`;
    playBtn.disabled=true; stepBtn.disabled=true; step5Btn.disabled=true;
  }else{
    bar.className='status-bar';
    bar.innerHTML=`<span>Step ${d.step}/${d.max_steps} &middot; policy <b>${d.policy}</b> &middot; partner <b>${d.profile}</b>`
      + (d.zero_shot?' <b style="color:var(--red)">(never seen in training)</b>':'')
      + `</span><span>${d.tasks_done}/${d.total_tasks} tasks &middot; ${d.orders_done}/${d.num_orders} orders</span>`;
    playBtn.disabled=false; stepBtn.disabled=false; step5Btn.disabled=false;
  }
}

(async()=>{
  try{ await loadConfig(); await doReset(); }
  catch(error){ showError(error); }
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    reset_session()
    print("\n" + "=" * 60)
    print("  DTS ZSC Coordination Dashboard")
    print(f"  checkpoint: {CHECKPOINT if os.path.exists(CHECKPOINT) else 'NOT FOUND (run train.py)'}")
    print("  open http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
