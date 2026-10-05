# What was built

An inventory of the DTS-ZSC deliverables: what each file is for, what each
capability does, how it was verified, and what the numbers came out at.

Companion documents:

* [`README.md`](../README.md) — how to run it, the MDP spec, the protocol
* [`PROJECT_STATUS.md`](../PROJECT_STATUS.md) — implementation report, results,
  the ablation, and the honest limitations
* [`docs/UI_GUIDE.md`](UI_GUIDE.md) — every dashboard control explained

**Scale:** 25 Python files, ~5,300 lines, 38 tests, 10 result charts,
3 diagrams, 8 commits.

---

## 1. What the system is

An AI agent decides, at every step, **which warehouse task runs next** and
**whether a human or a robot should do it** — for a human partner it has never
worked with before. Coordination is a sequential decision problem: the agent
reads generalised interaction signals rather than a profile of one person, and
re-plans on every step.

---

## 2. The code, file by file

### Environment — `env/`

| File | Lines | What it holds |
|---|---|---|
| `task_definitions.py` | 92 | The six-task order template with its dependency graph, and `Order` — arrival step, priority, deadline, slack |
| `workers.py` | 116 | `AIWorker` plus the protocol every worker implements, so the framework scales past one human and one robot |
| `collaborative_env.py` | 448 | The Gymnasium environment: observation, reward, the legality mask, and the step loop |

The environment owns three things worth calling out:

* **A WAIT action.** The agent can decline to assign. Without it, it must act
  every step and gets punished for situations it cannot control.
* **The legality mask.** `legal_action_mask()` is the single source of truth;
  the policy reads it rather than re-deriving it, so the two cannot disagree.
* **Reassignment.** An in-progress task can be taken off one worker and given
  to another, forfeiting the progress made, charged in proportion to what was
  lost.

### Human simulation — `human/simulated_human.py` (244 lines)

Ten partner types along six behavioural axes — speed, error rate, fatigue,
response latency, speed jitter, and a difficulty-dependent strategy bias. Five
are used for training, five are held out and never sampled during it.

### Agent — `agent/`

| File | Lines | What it holds |
|---|---|---|
| `ppo_agent.py` | 317 | Network, rollout buffer, PPO update, checkpoint I/O with a shape guard |
| `human_observer.py` | 138 | The Human Interaction Abstraction Layer — smoothed speed, error rate, completion ratio, trend |
| `curriculum.py` | 60 | Partner sampling weighted towards the ones the agent is currently worst at |

### Shared machinery — `utils/`

| File | Lines | What it holds |
|---|---|---|
| `metrics.py` | 204 | `EpisodeStats`, the Coordination Efficiency Score, per-profile table, zero-shot report |
| `episode.py` | 98 | `EpisodeTracker` — the one place per-episode statistics are assembled |
| `runner.py` | 103 | The shared episode and evaluation loop |
| `stats.py` | 105 | Bootstrap intervals and paired significance tests |
| `analytics.py` | 419 | Every chart |

### Entry points

| File | Lines | What it does |
|---|---|---|
| `train.py` | 321 | Training, with curriculum and ablation flags |
| `evaluate.py` | 281 | Per-partner results, zero-shot report, baselines, significance |
| `ablation.py` | 179 | Component study across matched training arms |
| `baselines.py` | 173 | Static, greedy and random policies |
| `plot_training.py` | 36 | CLI over the analytics module |
| `app.py` | 1,125 | The interactive coordination dashboard |
| `scripts/make_diagrams.py` | 206 | Renders the three documentation diagrams |
| `scripts/diagram_style.py` | 130 | Shared palette and drawing primitives |
| `tests/test_dts_zsc.py` | 542 | 38 correctness tests |

---

## 3. Capabilities, and where each lives

| Capability | Where |
|---|---|
| Dynamic task sequencing | `collaborative_env.py` action space; re-decided every step |
| Concurrent orders with deadlines | `task_definitions.Order`, staggered arrival |
| Adaptive prioritisation | order priority and slack in the observation and reward |
| Zero-shot coordination | `TRAIN_PROFILES` / `TEST_PROFILES` split, enforced at reset |
| Human interaction abstraction | `human_observer.py` |
| Prioritise, defer **and reassign** | WAIT action + takeover branch in `step()` |
| Scaling to N workers, N orders | `workers.py` protocol; `obs_dim_for` / `action_dim_for` |
| Interactive dashboard | `app.py` |
| Task dependency graph visualisation | live SVG DAG per order in `app.py` |
| Human behaviour simulation | `simulated_human.py`, six axes |
| Performance analytics | `analytics.py`, 10 charts |
| Statistical significance | `stats.py`, paired bootstrap |
| Component ablation | `ablation.py`, five matched arms |
| Curriculum sampling | `curriculum.py` |

---

## 4. Results

500 evaluation episodes per policy (50 per partner × 10 partners), identical
seeds, so the comparisons are paired.

| Policy | CES | Reward | Solve rate | ZSC gap | vs PPO |
|---|---|---|---|---|---|
| **PPO (learned)** | **0.690** | **+121.3** | **99.8%** | **−0.007** | — |
| greedy heuristic | 0.665 | +117.9 | 91.2% | −0.027 | +0.024, p<0.001 |
| partner specialist | 0.653 | +113.8 | 86.8% | −0.037 | +0.037, p<0.001 |
| static workflow | 0.482 | +79.2 | 76.0% | +0.003 | +0.208, p<0.001 |
| random | 0.444 | +57.1 | 56.2% | +0.031 | +0.246, p<0.001 |

Three things this shows:

* **The learned policy beats a fixed workflow by 0.208 CES** — the comparison
  the project abstract actually makes — and beats a strong informed heuristic
  by a smaller but statistically real margin.
* **No zero-shot penalty.** Held-out partners score 0.693 against 0.686 for
  familiar ones. Read that as "no measurable penalty", not as better.
* **The specialist fails where it matters.** An agent trained on one partner
  scores 0.345 on `novice` with a **0% solve rate**, against the zero-shot
  agent's 0.604 and 100%. That is the abstract's "behavioural model trained on
  historical interaction data", and it is what zero-shot is for.

### Component ablation

Five arms, matched 250k-step budgets and seeds, 40 episodes per partner.

| Arm | CES | Solve rate | Verdict |
|---|---|---|---|
| curriculum | 0.697 | 100.0% | helps, p<0.001 |
| blind partner | 0.691 | 99.8% | helps, p<0.001 |
| no observer | 0.688 | 99.2% | helps, p<0.001 |
| full | 0.682 | 98.5% | reference |
| no reassignment | 0.679 | 93.5% | no overall effect (p=0.61) |

Two findings worth reading carefully:

* **Reassignment looks neutral on average but is decisive on the hard case** —
  0.598 on `novice` with it, 0.428 without.
* **Removing the partner signals slightly improves the score.** This does not
  support the abstraction layer as an efficiency claim. The adaptive behaviour
  survives untouched (a blind agent still delegates 41.7% to an expert and 8.8%
  to a novice), because a slower worker simply stays busy longer and the policy
  reads speed from that. Reported as a negative result rather than smoothed
  over; see `PROJECT_STATUS.md` §2b.

---

## 5. Documentation and figures

| Artefact | What it is |
|---|---|
| `docs/images/architecture.png` | System architecture — environment, agent, outcomes |
| `docs/images/usecase.png` | UML use cases: three actors, eight cases, include/extend |
| `docs/images/activity.png` | UML activity: one episode, matching the real loop order |
| `results/*.png` | 10 analytics charts — training, coordination, optimisation, zero-shot gap, per-partner CES, policy comparison, adaptive allocation, specialist vs general, and two ablation charts |
| `results/evaluation.json` | Full evaluation output, machine-readable |
| `results/ablation.json` | Full ablation output |
| `results/episodes.csv` | Per-episode records for every policy |

All three diagrams regenerate with `python scripts/make_diagrams.py`.

---

## 6. How it was verified

**38 automated tests** (`python tests/test_dts_zsc.py`), targeting the failure
modes that make results untrustworthy rather than just line coverage:

* the action mask is brute-forced against the environment for every action in
  every visited state;
* held-out profiles are asserted never to leak into training over 200 resets;
* every held-out profile is asserted behaviourally distinct from a training one;
* evaluation is asserted reproducible from its seed, and non-reproducible
  without it;
* paired statistics refuse unaligned samples rather than comparing them quietly.

Beyond the suite, the diagrams were checked claim by claim against the running
code — 21 assertions covering the loop order, the reward terms, the termination
conditions and the held-out split — and the shareable archive was extracted to
a clean directory and run end to end.

---

## 7. Change history

| Commit | What it did |
|---|---|
| `45b700f` | The framework: environment, PPO agent, observer, metrics, baselines, dashboard, analytics, first test suite |
| `cad8a4c` | Closed two gaps found by reading the project abstract against the code — reassignment, and a partner-specialist baseline |
| `9f8911f` | Bootstrap intervals and paired significance; five-arm component ablation; fixed `--baselines` being a no-op without `--compare` |
| `bff8e2c` | Architecture diagram |
| `f027274` | Redrew it at presentation quality |
| `a728303` | Use case and activity diagrams |
| `fd0f8be` | Corrected the activity diagram to the real loop order |
| `4e3cb5b` | Robot utilisation reporting and selectable order urgency |

---

## 8. What this does not do

Stated plainly because a reviewer will ask:

* **No human subjects.** Partners are simulated, and the simulator is ours.
  The distance between "handles our five held-out profiles" and "handles a real
  warehouse worker" is large and unmeasured.
* **The abstraction layer is not shown to be necessary** (§4).
* **Every ablation arm is a single seed.** The intervals cover evaluation
  episodes, not training runs.
* **One scenario size was trained.** The framework is configurable and a
  3-order / 4-worker setup is tested for correctness, but every reported number
  is for 2 orders with 1 robot and 1 human.
* **Reassignment is free of human cost.** Taking a task off a person is modelled
  as pure progress loss; a real worker would find repeated preemption
  demoralising.
