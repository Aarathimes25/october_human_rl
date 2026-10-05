"""
Correctness tests for DTS-ZSC.

Run directly:
    python tests/test_dts_zsc.py

or under pytest:
    pytest tests/test_dts_zsc.py -q

These target the specific failure modes that made the earlier revision's
results untrustworthy:
  * the action mask silently disagreeing with the environment
  * task completion never actually being counted
  * held-out profiles that were behaviourally identical to training ones
  * evaluation seeds being discarded by a second reset
"""

import os
import sys
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agent.curriculum import PartnerCurriculum
from agent.human_observer import HumanBehaviorObserver, extra_obs_dim
from agent.ppo_agent import PPOAgent
from baselines import RandomPolicy, StaticPolicy, GreedyPolicy, make_baseline
from env.collaborative_env import (CollaborativeTaskEnv, obs_dim_for,
                                   action_dim_for)
from env.task_definitions import ORDER_TEMPLATE, TASKS_PER_ORDER
from human.simulated_human import (SimulatedHuman, HUMAN_PROFILES,
                                   TRAIN_PROFILES, TEST_PROFILES)
from utils.metrics import CoordinationMetrics, EpisodeStats
from utils.runner import run_episode, evaluate_policy
from utils.stats import bootstrap_ci, paired_diff, episode_ces


# ── environment ───────────────────────────────────────────────────────── #
def test_spaces_match_helpers():
    env = CollaborativeTaskEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (obs_dim_for(),)
    assert env.action_dim == action_dim_for()
    assert env.observation_space.contains(obs)
    assert env.wait_action == env.action_dim - 1


def test_observation_stays_inside_declared_box():
    env = CollaborativeTaskEnv()
    obs, info = env.reset(seed=11)
    for _ in range(200):
        legal = np.flatnonzero(info["action_mask"])
        obs, _, term, trunc, info = env.step(int(np.random.default_rng(_).choice(legal)))
        assert env.observation_space.contains(obs), "observation left the declared Box"
        if term or trunc:
            obs, info = env.reset()


def test_mask_exactly_matches_environment_legality():
    """
    Brute force: for every state visited, step a deep copy of the environment
    with every action and confirm the mask predicted the conflict flag.
    """
    env = CollaborativeTaskEnv()
    for trial in range(12):
        obs, info = env.reset(seed=trial)
        for t in range(8):
            mask = info["action_mask"]
            for a in range(env.action_dim):
                probe = copy.deepcopy(env)
                *_, pinfo = probe.step(a)
                assert bool(mask[a]) == (not pinfo["conflict"]), (
                    f"mask/env disagree on action {a} at trial {trial} step {t}")
            legal = np.flatnonzero(mask)
            obs, _, term, trunc, info = env.step(
                int(np.random.default_rng(trial * 97 + t).choice(legal)))
            if term or trunc:
                break


def test_wait_is_always_legal_and_never_a_conflict():
    env = CollaborativeTaskEnv()
    _, info = env.reset(seed=1)
    for _ in range(env.max_steps):
        assert info["action_mask"][env.wait_action]
        _, _, term, trunc, info = env.step(env.wait_action)
        assert not info["conflict"]
        if term or trunc:
            break


def test_human_can_be_assigned_while_ai_is_busy():
    """The bug that made parallel work impossible in the earlier revision."""
    env = CollaborativeTaskEnv()
    _, info = env.reset(seed=4)
    ai, human = 0, 1
    env.step(ai * env.num_global_tasks + 0)            # AI takes task 0
    _, _, _, _, info = env.step(human * env.num_global_tasks + 1)  # human takes task 1
    assert not info["conflict"], "assigning the human while the AI works was rejected"
    assert info["num_busy"] == 2, "the two workers are not working in parallel"


def test_completion_is_actually_counted():
    """`tasks_done` must rise as work finishes — it was hard-wired to 0 before."""
    env = CollaborativeTaskEnv(human_profile=HUMAN_PROFILES["expert"])
    _, info = env.reset(seed=2)
    policy = GreedyPolicy(env, seed=0)
    seen_progress = False
    for _ in range(env.max_steps):
        a, _, _ = policy.select_action(None, info)
        _, _, term, trunc, info = env.step(a)
        if info["tasks_done"] > 0:
            seen_progress = True
        if term or trunc:
            break
    assert seen_progress, "tasks_done never moved off zero"
    assert info["tasks_done"] == info["total_tasks"], "greedy failed to finish an easy scenario"
    assert info["orders_done"] == info["num_orders"]


def test_idle_penalty_does_not_fire_when_nothing_is_available():
    """Waiting while genuinely blocked must not be punished."""
    env = CollaborativeTaskEnv(num_orders=1)
    _, info = env.reset(seed=0)
    env.step(0 * env.num_global_tasks + 0)     # AI on fetch A
    _, _, _, _, info = env.step(1 * env.num_global_tasks + 1)   # human on fetch B
    assert not info["eligible_tasks"], "expected the team to be fully occupied"
    _, reward, _, _, info = env.step(env.wait_action)
    # only the time penalty and the sync bonus should apply
    assert reward > 0, f"waiting while blocked was penalised (reward {reward})"


def test_reassignment_moves_a_task_between_workers():
    """The methodology promises the agent can reassign, not only start and defer."""
    env = CollaborativeTaskEnv(allow_reassignment=True)
    _, info = env.reset(seed=0)
    ai, human = 0, 1
    N = env.num_global_tasks

    env.step(ai * N + 0)                              # AI starts task 0
    assert info is not None
    _, _, _, _, info = env.step(env.wait_action)      # let some progress accrue
    assert info["workers"][ai]["current_task"] == 0

    takeover = human * N + 0
    assert info["action_mask"][takeover], "taking the task over should be legal"
    _, _, _, _, info = env.step(takeover)

    assert not info["conflict"]
    assert info["reassigned"] and info["reassignments"] == 1
    assert info["workers"][ai]["current_task"] is None, "old holder was not released"
    assert info["workers"][human]["current_task"] == 0, "new holder did not pick it up"
    assert info["workers"][human]["progress"] == 0.0, "progress should not carry over"


def test_reassignment_costs_more_the_later_it_happens():
    """
    Bailing out early is cheap; bailing out near the end is expensive.

    The takeover reward is compared against *waiting from the identical state*
    (via a deep copy), so the two branches differ only by the consequences of
    the takeover — comparing raw step rewards would be swamped by the other
    reward terms.
    """
    def cost_of_taking_over(wait_steps):
        env = CollaborativeTaskEnv(human_profile=HUMAN_PROFILES["average"])
        env.reset(seed=3)
        N = env.num_global_tasks
        env.step(0 * N + 0)                       # AI starts task 0
        for _ in range(wait_steps):
            env.step(env.wait_action)
        progress = env.workers[0].progress

        took = copy.deepcopy(env)
        _, r_take, *_ = took.step(1 * N + 0)      # human takes it over
        left = copy.deepcopy(env)
        _, r_wait, *_ = left.step(env.wait_action)
        return progress, r_take - r_wait

    p_early, cost_early = cost_of_taking_over(0)
    p_late,  cost_late  = cost_of_taking_over(2)

    assert p_late > p_early, "the later takeover should interrupt more progress"
    assert cost_late < cost_early, (
        f"taking over at {p_late:.0%} progress (cost {cost_late:+.2f}) should be "
        f"worse than at {p_early:.0%} (cost {cost_early:+.2f})")


def test_reassignment_can_be_disabled():
    env = CollaborativeTaskEnv(allow_reassignment=False)
    _, info = env.reset(seed=0)
    N = env.num_global_tasks
    env.step(0 * N + 0)
    _, _, _, _, info = env.step(1 * N + 0)
    assert info["conflict"], "takeover should be illegal when the ablation is on"
    assert info["reassignments"] == 0
    assert not info["reassignable_tasks"]


def test_a_worker_cannot_take_a_task_from_itself():
    env = CollaborativeTaskEnv()
    _, info = env.reset(seed=0)
    N = env.num_global_tasks
    _, _, _, _, info = env.step(0 * N + 0)
    assert not info["action_mask"][0 * N + 0], "self-takeover must stay illegal"


def test_environment_accepts_a_partner_override():
    """The curriculum names the partner for each episode via reset options."""
    env = CollaborativeTaskEnv()
    for want in TRAIN_PROFILES:
        _, info = env.reset(seed=0, options={"profile": want})
        assert info["human_metrics"]["profile"] == want
    try:
        env.reset(seed=0, options={"profile": "not_a_profile"})
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown profile should be rejected")


def test_order_priorities_can_be_pinned():
    """Urgency drives the deadline, so pinning it pins the time pressure."""
    env = CollaborativeTaskEnv(num_orders=2, order_priorities=[0.95, 0.20])
    _, info = env.reset(seed=0)
    urgent, routine = info["orders"]

    assert abs(urgent["priority"] - 0.95) < 1e-9
    assert abs(routine["priority"] - 0.20) < 1e-9
    assert urgent["deadline"] < routine["deadline"], \
        "a more urgent order must get the tighter deadline"

    # and it survives a reseed, unlike the sampled default
    for seed in range(4):
        _, info = env.reset(seed=seed)
        assert abs(info["orders"][0]["priority"] - 0.95) < 1e-9


def test_every_worker_reports_how_busy_it_was():
    """The robot's speed and error rate are constants; utilisation is not."""
    env = CollaborativeTaskEnv(human_profile=HUMAN_PROFILES["average"])
    _, info = env.reset(seed=2)
    policy = GreedyPolicy(env, seed=0)
    for _ in range(12):
        action, _, _ = policy.select_action(None, info)
        _, _, term, trunc, info = env.step(action)
        if term or trunc:
            break

    for worker in info["workers"]:
        assert "busy_steps" in worker, f"{worker['name']} reports no busy time"
        assert 0 <= worker["busy_steps"] <= info["step"]
    assert any(w["busy_steps"] > 0 for w in info["workers"] if w["kind"] == "ai"), \
        "the robot did work but recorded none"


def test_scaling_to_more_workers_and_orders():
    env = CollaborativeTaskEnv(num_orders=3, num_humans=2, num_ai=2, max_steps=90)
    obs, info = env.reset(seed=0)
    assert obs.shape == (obs_dim_for(3, 4),)
    assert env.action_dim == action_dim_for(3, 4)
    assert len(info["workers"]) == 4
    policy = GreedyPolicy(env, seed=0)
    hbo = HumanBehaviorObserver(num_humans=2, total_tasks=env.num_global_tasks)
    stats, _ = run_episode(env, policy, hbo, seed=0)
    assert stats.conflict_count == 0
    assert stats.total_tasks == 18


# ── human simulation ──────────────────────────────────────────────────── #
def _time_all_tasks(profile_name: str, seed: int = 7):
    human = SimulatedHuman(HUMAN_PROFILES[profile_name], seed=seed)
    times = []
    for spec in ORDER_TEMPLATE:
        human.assign(spec.idx, spec)
        n = 1
        while human.step(1.0) is None and n < 500:
            n += 1
        times.append(n)
    return times


def test_held_out_profiles_are_behaviourally_distinct():
    """
    `easy_first` used to differ from `average` only by an unread field, making
    it an exact behavioural duplicate of a training profile.
    """
    baseline = _time_all_tasks("average")
    for name in TEST_PROFILES:
        other = _time_all_tasks(name)
        assert other != baseline, (
            f"held-out profile '{name}' behaves identically to a training profile")


def test_strategy_bias_changes_the_difficulty_ordering():
    easy = _time_all_tasks("easy_first")
    hard = _time_all_tasks("hard_first")
    pack = 2                                  # the hardest task, difficulty 0.6
    label = 3                                 # an easy task, difficulty 0.2
    assert easy[pack] / max(easy[label], 1) > hard[pack] / max(hard[label], 1), \
        "easy_first should be relatively worse on hard tasks than hard_first"


def test_response_latency_delays_the_start():
    human = SimulatedHuman(HUMAN_PROFILES["hesitant"], seed=0)
    human.assign(0, ORDER_TEMPLATE[0])
    assert human.is_waiting
    for _ in range(3):
        assert human.step(1.0) is None
    assert not human.is_waiting, "worker never started after its latency elapsed"


def test_train_and_test_profiles_never_overlap():
    assert not (set(TRAIN_PROFILES) & set(TEST_PROFILES))
    env = CollaborativeTaskEnv()
    sampled = set()
    for s in range(200):
        _, info = env.reset(seed=s)
        sampled.add(info["human_metrics"]["profile"])
    assert sampled <= set(TRAIN_PROFILES), f"held-out profile leaked into training: {sampled}"


# ── agent ─────────────────────────────────────────────────────────────── #
def test_masked_agent_never_produces_a_conflict():
    env = CollaborativeTaskEnv()
    agent = PPOAgent(obs_dim=obs_dim_for() + extra_obs_dim(1),
                     action_dim=action_dim_for(), rollout_len=128)
    hbo = HumanBehaviorObserver(num_humans=1, total_tasks=env.num_global_tasks)
    for seed in range(5):
        stats, _ = run_episode(env, agent, hbo, seed=seed, deterministic=False)
        assert stats.conflict_count == 0


def test_ppo_update_produces_finite_losses():
    env = CollaborativeTaskEnv()
    agent = PPOAgent(obs_dim=obs_dim_for() + extra_obs_dim(1),
                     action_dim=action_dim_for(), rollout_len=128, minibatch=32)
    hbo = HumanBehaviorObserver(num_humans=1, total_tasks=env.num_global_tasks)
    obs, info = env.reset(seed=0)
    hbo.reset()
    obs = hbo.enhance(obs)
    for _ in range(128):
        a, lp, v = agent.select_action(obs, info)
        raw, r, term, trunc, nxt = env.step(a)
        hbo.update(nxt)
        agent.store(obs, a, r * 0.1, v, lp, float(term or trunc), info)
        obs, info = hbo.enhance(raw), nxt
        if term or trunc:
            raw, info = env.reset()
            hbo.reset()
            obs = hbo.enhance(raw)
    losses = agent.update(obs, info)
    for key, value in losses.items():
        assert np.isfinite(value), f"{key} is not finite"
    assert agent.buffer.ptr == 0, "buffer was not cleared after the update"


def test_checkpoint_round_trip_and_shape_guard(tmp_path=None):
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "dtszsc_test_agent.pt")
    a = PPOAgent(obs_dim=obs_dim_for() + extra_obs_dim(1), action_dim=action_dim_for())
    a.save(path)
    b = PPOAgent(obs_dim=obs_dim_for() + extra_obs_dim(1), action_dim=action_dim_for())
    b.load(path)
    for p, q in zip(a.net.parameters(), b.net.parameters()):
        assert np_allclose(p, q)
    mismatched = PPOAgent(obs_dim=obs_dim_for(3, 4) + extra_obs_dim(2),
                          action_dim=action_dim_for(3, 4))
    try:
        mismatched.load(path)
    except ValueError:
        pass
    else:
        raise AssertionError("loading into a mismatched configuration should fail")
    os.remove(path)


def np_allclose(p, q):
    return bool((p.detach() - q.detach()).abs().max().item() < 1e-9)


# ── abstraction-layer ablation ────────────────────────────────────────── #
def test_observer_ablation_zeros_features_without_changing_the_dimension():
    """
    The ablation must isolate the *information*, not the network size — so the
    feature block stays the same length and only its contents go to zero.
    """
    live = HumanBehaviorObserver(num_humans=1, total_tasks=12, enabled=True)
    dead = HumanBehaviorObserver(num_humans=1, total_tasks=12, enabled=False)
    sample = {"human_metrics_list": [{"speed": 0.9, "error_rate": 0.2,
                                      "tasks_done": 3}]}
    for _ in range(6):
        live.update(sample)
        dead.update(sample)
        live.record_assignment()
        dead.record_assignment()

    assert live.extra_features().shape == dead.extra_features().shape
    assert live.extra_features().shape == (extra_obs_dim(1),)
    assert np.any(live.extra_features() != 0.0), "live observer produced nothing"
    assert np.all(dead.extra_features() == 0.0), "ablated observer leaked signal"


def test_blind_partner_hides_capability_but_keeps_scheduling_state():
    """`no_observer` leaves the raw partner signals visible; this removes them."""
    from env.workers import WORKER_FEATURES, CAPABILITY_FEATURE_SLICE

    for blind in (False, True):
        env = CollaborativeTaskEnv(blind_partner=blind)
        env.reset(seed=0)
        env.step(0 * env.num_global_tasks + 0)
        obs, *_ = env.step(1 * env.num_global_tasks + 1)

        base = env.num_orders * TASKS_PER_ORDER * 6
        blocks = [obs[base + i * WORKER_FEATURES: base + (i + 1) * WORKER_FEATURES]
                  for i in range(env.num_workers)]
        caps = np.concatenate([b[CAPABILITY_FEATURE_SLICE] for b in blocks])
        scheduling = np.concatenate([b[:3] for b in blocks])

        assert obs.shape == env.observation_space.shape, "dimension must not change"
        assert np.any(scheduling != 0.0), "scheduling state must survive"
        if blind:
            assert np.all(caps == 0.0), "capability signals leaked while blind"
        else:
            assert np.any(caps != 0.0), "capability signals missing when not blind"


def test_checkpoint_records_its_training_configuration():
    """Evaluation reads this to avoid pairing an ablated agent with a live observer."""
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "dtszsc_cfg_test.pt")
    saved = PPOAgent(obs_dim=obs_dim_for() + extra_obs_dim(1),
                     action_dim=action_dim_for())
    saved.save(path, config={"no_observer": True, "no_reassignment": False})

    loaded = PPOAgent(obs_dim=obs_dim_for() + extra_obs_dim(1),
                      action_dim=action_dim_for())
    loaded.load(path)
    assert loaded.config == {"no_observer": True, "no_reassignment": False}
    os.remove(path)


# ── curriculum ────────────────────────────────────────────────────────── #
# CES roughly as observed in training, with novice far behind the rest.
CES_BY_PARTNER = {"average": 0.72, "fast": 0.73, "slow": 0.67,
                  "expert": 0.70, "novice": 0.40}


def _warmed_curriculum(**kwargs) -> PartnerCurriculum:
    cur = PartnerCurriculum(list(TRAIN_PROFILES), warmup=2, **kwargs)
    for _ in range(6):
        for profile, ces in CES_BY_PARTNER.items():
            cur.update(profile, ces)
    return cur


def test_curriculum_is_uniform_until_every_partner_is_warmed_up():
    pool = list(TRAIN_PROFILES)
    cur = PartnerCurriculum(pool, warmup=3)
    assert np.allclose(cur.weights(), 1.0 / len(pool))
    for _ in range(2):                       # still under the warm-up threshold
        for profile in pool:
            cur.update(profile, 0.5)
    assert np.allclose(cur.weights(), 1.0 / len(pool))


def test_curriculum_favours_weak_partners_but_respects_the_floor():
    pool = list(TRAIN_PROFILES)
    weights = _warmed_curriculum(min_share=0.4).weights()

    assert abs(weights.sum() - 1.0) < 1e-9, "weights must be a distribution"
    assert weights[pool.index("novice")] == weights.max()
    assert weights.min() >= 0.4 / len(pool) - 1e-9, "a mastered partner was starved"
    assert weights.max() < 0.5, f"curriculum collapsed: {weights.max():.2f}"


def test_curriculum_sampling_matches_its_weights():
    cur = _warmed_curriculum(min_share=0.4)
    rng = np.random.default_rng(0)
    draws = [cur.sample(rng) for _ in range(4000)]
    counts = {profile: draws.count(profile) for profile in TRAIN_PROFILES}
    assert all(counts.values()), "every partner must still appear"
    assert counts["novice"] == max(counts.values())


# ── statistics ────────────────────────────────────────────────────────── #
def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    rng = np.random.default_rng(0)
    sample = rng.normal(0.6, 0.1, size=60)
    first, repeat = bootstrap_ci(sample, seed=5), bootstrap_ci(sample, seed=5)

    assert first.lo <= first.mean <= first.hi
    assert (first.lo, first.hi) == (repeat.lo, repeat.hi), "seed must be honoured"
    assert bootstrap_ci([], seed=0).n == 0
    only_one = bootstrap_ci([0.4], seed=0)
    assert only_one.lo == only_one.hi == 0.4


def test_paired_test_detects_a_real_effect_and_ignores_noise():
    rng = np.random.default_rng(1)
    base = rng.normal(0.60, 0.08, size=120)
    better = base + rng.normal(0.03, 0.01, size=120)
    noise = base + rng.normal(0.00, 0.05, size=120)

    real = paired_diff(better, base, "better", "base", seed=0)
    null = paired_diff(noise, base, "noise", "base", seed=0)

    assert real.significant and real.diff > 0 and real.p_value < 0.05
    assert real.lo > 0, "CI for a real gain should exclude zero"
    assert not null.significant
    assert null.lo < 0 < null.hi, "CI for noise should straddle zero"


def test_paired_test_refuses_unaligned_samples():
    try:
        paired_diff([1.0, 2.0, 3.0], [1.0, 2.0])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched lengths must not be silently compared")


def test_episode_ces_aligns_across_policies():
    """Pairing is only valid if two policies produce the same episode order."""
    profiles = ["average", "novice"]
    greedy = evaluate_policy(lambda e: GreedyPolicy(e, seed=0), profiles,
                             episodes=4, seed=77)
    static = evaluate_policy(lambda e: StaticPolicy(e, seed=0), profiles,
                             episodes=4, seed=77)

    assert len(episode_ces(greedy)) == len(episode_ces(static)) == 8
    assert ([s.profile for s in greedy.history]
            == [s.profile for s in static.history])


# ── metrics ───────────────────────────────────────────────────────────── #
def _stats(**kw):
    base = dict(profile="average", steps=30, total_reward=100.0,
                tasks_completed=12, total_tasks=12, orders_completed=2,
                orders_on_time=2, num_orders=2, human_tasks=5, ai_tasks=7,
                role_switches=4, parallel_steps=15, idle_steps=0,
                conflict_count=0, wait_actions=10, reassignments=0,
                completion_time=30.0, max_steps=70, solved=True)
    base.update(kw)
    return EpisodeStats(**base)


def test_ces_arithmetic():
    perfect = _stats(completion_time=0.0, parallel_steps=30)
    assert abs(CoordinationMetrics.ces(perfect) - 0.90) < 1e-9
    nothing = _stats(tasks_completed=0, orders_on_time=0, parallel_steps=0,
                     completion_time=70.0, solved=False)
    assert abs(CoordinationMetrics.ces(nothing)) < 1e-9
    conflicted = _stats(tasks_completed=0, orders_on_time=0, parallel_steps=0,
                        completion_time=70.0, conflict_count=30, solved=False)
    assert CoordinationMetrics.ces(conflicted) < 0


def test_zsc_report_splits_seen_from_held_out():
    m = CoordinationMetrics(window=10 ** 6)
    for p in TRAIN_PROFILES:
        m.record(_stats(profile=p, parallel_steps=15))
    for p in TEST_PROFILES:
        m.record(_stats(profile=p, parallel_steps=0))     # deliberately worse
    rep = m.zsc_report(TRAIN_PROFILES, TEST_PROFILES)
    assert set(rep["seen"]) == set(TRAIN_PROFILES)
    assert set(rep["held_out"]) == set(TEST_PROFILES)
    assert rep["zsc_gap"] > 0
    assert abs(rep["zsc_gap"] - (rep["seen_ces"] - rep["held_out_ces"])) < 1e-12


# ── baselines and the runner ──────────────────────────────────────────── #
def test_baselines_never_take_an_illegal_action():
    for name in ("random", "static", "greedy"):
        m = evaluate_policy(lambda env, n=name: make_baseline(n, env, seed=0),
                            ["average", "novice"], episodes=4, seed=0)
        assert m.conflict_rate(10 ** 9) == 0.0, f"{name} produced conflicts"


def test_greedy_beats_static_on_average():
    g = evaluate_policy(lambda e: GreedyPolicy(e, seed=0), ["average"],
                        episodes=10, seed=0)
    s = evaluate_policy(lambda e: StaticPolicy(e, seed=0), ["average"],
                        episodes=10, seed=0)
    assert g.mean_ces(10 ** 9) > s.mean_ces(10 ** 9), \
        "the informed heuristic should beat the fixed workflow"


def test_evaluation_is_reproducible():
    """The seed used to be discarded by a second reset inside the loop."""
    runs = [evaluate_policy(lambda e: GreedyPolicy(e, seed=0), ["average"],
                            episodes=6, seed=123).mean_reward(10 ** 9)
            for _ in range(2)]
    assert abs(runs[0] - runs[1]) < 1e-9, "identical seeds gave different results"

    different = evaluate_policy(lambda e: GreedyPolicy(e, seed=0), ["average"],
                                episodes=6, seed=999).mean_reward(10 ** 9)
    assert abs(runs[0] - different) > 1e-9, "the seed had no effect at all"


# ── dashboard ─────────────────────────────────────────────────────────── #
def test_dashboard_api():
    import app as dash
    client = dash.app.test_client()

    cfg = client.get("/api/config").get_json()
    assert set(cfg["seen_profiles"]) == set(TRAIN_PROFILES)
    assert set(cfg["held_out_profiles"]) == set(TEST_PROFILES)
    assert cfg["research"]["available"]
    assert len(cfg["research"]["comparisons"]) == 3
    assert len(cfg["research"]["evidence"]) == 5
    assert len(cfg["priorities"]) == 7
    assert "custom" in {p["key"] for p in cfg["priorities"]}
    assert cfg["max_orders"] == 6

    r = client.post("/api/reset", json={"profile": "hesitant", "policy": "greedy",
                                        "num_orders": 2, "num_ai": 1, "num_humans": 1})
    assert r.status_code == 200
    d = client.post("/api/step", json={"count": 200}).get_json()
    assert d["done"] and d["conflicts"] == 0
    assert d["explanations"]
    assert {"decision", "why", "role_change", "trigger", "source"} <= set(d["explanations"][0])
    assert d["observer"]["enabled"] and d["observer"]["humans"]
    assert len(d["observer"]["feature_order"]) == 5
    assert {"basis", "ready", "active"} <= set(d["sequencing"])
    assert len(d["orders"]) == 2
    for order in d["orders"]:
        assert 1 <= order["priority_rank"] <= 2
        assert len(order["tasks"]) == TASKS_PER_ORDER
        for t in order["tasks"]:
            assert t["status"] in ("pending", "active", "done")
            assert 0.0 <= t["progress"] <= 1.0
    for worker in d["workers"]:
        assert {"throughput", "completion_share", "utilisation"} <= set(worker)
    html = client.get("/").get_data(as_text=True)
    assert "—" not in html and "–" not in html
    assert "&mdash;" not in html and "&ndash;" not in html and "&#8212;" not in html
    for dashed_label in ("DTS-ZSC", "Zero-Shot", "zero-shot", "Human-AI",
                         "Human–AI", "Held-out", "held-out", "Observed-input"):
        assert dashed_label not in html
    for marker in ("buildDag", "dag-wrap", "o.tasks_done/6", "brand-kicker", "control-field",
                   "main-column", "right-rail", "renderResearch",
                   "renderExplanations", "renderObserver", "renderSequencing",
                   "Human Observer output", "Step sequencing basis",
                   "Performance comparisons", "Research evidence"):
        assert marker in html


def test_dashboard_exposed_scenario_matrix_completes():
    """Every baseline/workload/team selector combination must keep progressing."""
    import app as dash
    client = dash.app.test_client()

    for policy in ("greedy", "static", "random"):
        for num_orders in (1, 2, 3, 4, 5, 6):
            for num_ai, num_humans in ((1, 1), (1, 2), (2, 2)):
                reset = client.post("/api/reset", json={
                    "profile": "average", "policy": policy,
                    "num_orders": num_orders, "num_ai": num_ai,
                    "num_humans": num_humans, "priority": "mixed",
                })
                assert reset.status_code == 200
                initial = reset.get_json()
                expected_steps = (70 + max(0, num_orders - 2) * 20 if num_orders <= 3
                                  else 90 + (num_orders - 3) * 35)
                assert initial["max_steps"] == expected_steps
                result = client.post("/api/step", json={"count": 300}).get_json()
                assert result["solved"], (
                    f"{policy}, {num_orders} orders, {num_ai} AI + "
                    f"{num_humans} humans stopped at "
                    f"{result['tasks_done']}/{result['total_tasks']}")


def test_dashboard_custom_priority_sequence_drives_ranking():
    """An operator-supplied ranking must set the urgency and deadline order."""
    import app as dash
    client = dash.app.test_client()

    d = client.post("/api/reset", json={
        "profile": "average", "policy": "greedy", "num_orders": 6,
        "num_ai": 2, "num_humans": 2,
        "priority": "custom", "priority_order": "1,2,3,6,5,4",
    }).get_json()
    assert d["priority_order"] == [1, 2, 3, 6, 5, 4]

    by_number = {o["number"]: o for o in d["orders"]}
    for rank, number in enumerate([1, 2, 3, 6, 5, 4], 1):
        assert by_number[number]["priority_rank"] == rank

    # Urgency must fall monotonically along the operator's sequence, and the
    # deadline must tighten with it relative to the order's own arrival.
    ranked = [by_number[n] for n in [1, 2, 3, 6, 5, 4]]
    priorities = [o["priority"] for o in ranked]
    assert priorities == sorted(priorities, reverse=True)
    budgets = [o["deadline"] - o["arrival"] for o in ranked]
    assert budgets == sorted(budgets)

    final = client.post("/api/step", json={"count": 300}).get_json()
    assert final["solved"]
    # Each order announces its completion exactly once.
    finished = [line["text"] for line in final["log"] if "complete (" in line["text"]]
    assert len(finished) == len(set(finished))


def test_dashboard_rejects_malformed_priority_sequence():
    import app as dash
    client = dash.app.test_client()

    for payload, fragment in (
        ({"num_orders": 7}, "between 1 and 6"),
        ({"num_orders": 4, "priority": "custom", "priority_order": "1,2,3"}, "exactly once"),
        ({"num_orders": 4, "priority": "custom", "priority_order": "1,2,2,3"}, "exactly once"),
        ({"num_orders": 4, "priority": "custom", "priority_order": "x,y"}, "whole order numbers"),
    ):
        bad = client.post("/api/reset", json=payload)
        assert bad.status_code == 400 and fragment in bad.get_json()["error"]

    # An empty sequence is a valid request meaning "natural order 1..N".
    ok = client.post("/api/reset", json={"num_orders": 5, "priority": "custom",
                                         "priority_order": ""}).get_json()
    assert ok["priority_order"] == [1, 2, 3, 4, 5]


def test_dashboard_ppo_dimension_mismatch_uses_labelled_fallback():
    """Custom PPO scenarios must never run a silent, untrained network."""
    import tempfile
    import app as dash
    saved_dir = dash.SCENARIO_CHECKPOINT_DIR
    try:
        # An empty directory: no scenario model has been trained.
        with tempfile.TemporaryDirectory() as empty:
            dash.SCENARIO_CHECKPOINT_DIR = empty
            client = dash.app.test_client()
            result = client.post("/api/reset", json={
                "profile": "average", "policy": "ppo", "num_orders": 3,
                "num_ai": 2, "num_humans": 2, "priority": "urgent",
            }).get_json()
            assert result["requested_policy"] == "ppo"
            assert result["policy"] == "greedy (PPO fallback)"
            assert "no untrained model" in result["note"]
            final = client.post("/api/step", json={"count": 200}).get_json()
            assert final["solved"] and final["tasks_done"] == final["total_tasks"]
    finally:
        dash.SCENARIO_CHECKPOINT_DIR = saved_dir


def test_dashboard_loads_scenario_specific_ppo_checkpoint():
    """A model saved for a non-default scenario is run as PPO for that scenario."""
    import tempfile
    import app as dash
    saved_dir = dash.SCENARIO_CHECKPOINT_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dash.SCENARIO_CHECKPOINT_DIR = tmp
            path = dash.scenario_checkpoint(1, 1, 2)
            os.makedirs(os.path.dirname(path))
            PPOAgent(obs_dim=obs_dim_for(1, 3) + extra_obs_dim(2),
                     action_dim=action_dim_for(1, 3)).save(path)
            client = dash.app.test_client()
            result = client.post("/api/reset", json={
                "profile": "average", "policy": "ppo", "num_orders": 1,
                "num_ai": 1, "num_humans": 2,
            }).get_json()
            assert result["policy"] == "ppo"
            final = client.post("/api/step", json={"count": 5}).get_json()
            assert final["step"] == 5 and final["conflicts"] == 0
    finally:
        dash.SCENARIO_CHECKPOINT_DIR = saved_dir


def test_dashboard_default_scenario_still_uses_main_checkpoint():
    import app as dash
    assert dash.scenario_checkpoint(dash.DEFAULT_NUM_ORDERS, dash.DEFAULT_NUM_AI,
                                    dash.DEFAULT_NUM_HUMANS) == dash.CHECKPOINT


def test_dashboard_rejects_invalid_selector_payloads():
    import app as dash
    client = dash.app.test_client()
    bad = client.post("/api/reset", json={"profile": "does-not-exist"})
    assert bad.status_code == 400 and "unknown partner" in bad.get_json()["error"]
    bad_step = client.post("/api/step", json={"count": 0})
    assert bad_step.status_code == 400


# ── runner ────────────────────────────────────────────────────────────── #
def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print("-" * 70)
    print(f"{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        import traceback
        for name, exc in failures:
            print(f"\n--- {name} ---")
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        return 1
    return 0


if __name__ == "__main__":
    print("=" * 70)
    print("DTS-ZSC test suite")
    print("=" * 70)
    sys.exit(main())
