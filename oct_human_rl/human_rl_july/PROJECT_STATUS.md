# DTS-ZSC — Implementation Report

**Project:** Dynamic Task Sequencing for Zero-Shot Human–AI Coordination using Reinforcement Learning
**Team:** Aarathi Nair · Archana Ghadage · Prachi Mandhare
**Supervisor:** Prof. Tejashree Patil · **HoD:** Prof. Abhijeet More

**Report date:** 2 August 2026
**Status:** all 24 items of the work plan implemented and verified, plus two
gaps found by re-reading this project's own abstract against the code, plus a
component ablation and significance testing. 36/36 tests pass.

> This file replaces the earlier gap analysis. Section 6 lists every problem
> that analysis found and what was done about it; section 6b covers the two
> later gaps; section 2b reports the ablation, including a result that does
> not support one of the project's stated novelties.

---

## 1. What changed, in one paragraph

The earlier revision had the right architecture but could not learn: a single
condition in the environment made it illegal to give the human any work while
the AI robot was busy, and there was no way to say "wait", so the agent was
punished several times per step for situations it could not control. Task
completion was also never counted (a always-true expression made it a constant
zero), and one of the two held-out "zero-shot" partners was behaviourally
identical to a training partner. Those are fixed. The environment now supports
multiple concurrent orders with deadlines and urgency, any number of human and
robot workers, and five genuinely distinct held-out partner types. The agent
trains cleanly, beats a strong hand-written heuristic, and coordinates with
partners it has never met about as well as with familiar ones.

---

## 2. Results

Trained for 400,000 steps on CPU (~25 minutes). Evaluated over 50 episodes per
partner across 10 partner profiles, with fixed seeds.

### Learned policy vs. scripted baselines

500 episodes per policy (50 × 10 partner profiles), identical seeds, all
policies restricted to legal actions.

| Policy | CES | Reward | Steps | Tasks done | Solved | On time | Sync | Reassign/ep |
|---|---|---|---|---|---|---|---|---|
| **PPO (zero-shot)** | **0.690** | **+121.3** | **32.0** | **99.9%** | **99.8%** | 84.7% | 0.412 | 0.30 |
| greedy heuristic | 0.665 | +117.9 | 35.6 | 96.6% | 91.2% | **87.4%** | 0.361 | 0.57 |
| specialist (one partner) | 0.653 | +113.8 | 36.8 | 95.3% | 86.8% | 83.6% | 0.382 | 0.11 |
| static workflow | 0.482 | +79.2 | 51.9 | 88.8% | 76.0% | 53.1% | 0.088 | 0.00 |
| random | 0.444 | +57.1 | 58.5 | 88.6% | 56.2% | 32.9% | 0.236 | 16.65 |

The learned policy is **+0.208 CES over the static predefined workflow** (+43%
relative) — the comparison the abstract actually makes — **+0.024 over a strong
informed heuristic**, and **+0.037 over a partner-specific agent**. It finishes
orders in 32.0 steps against the static workflow's 51.9, and completes
**99.8% of episodes** against 76.0%.

Greedy edges it on on-time delivery (87.4% vs 84.7%) while losing on everything
else — worth stating rather than hiding. Random's 16.65 reassignments per
episode is the capability being abused: it thrashes, which is why it now scores
below the static workflow.

### Zero-shot generalisation

| Partner group | Mean CES | Mean reward | Solve rate |
|---|---|---|---|
| seen during training (5 profiles) | 0.686 | +121.3 | 100.0% |
| **held out, never trained on (5 profiles)** | **0.693** | **+121.3** | **99.6%** |
| gap | **−0.007** (−1.0% relative) | | |

Per-partner CES:

| Seen | CES | | Held out (zero-shot) | CES |
|---|---|---|---|---|
| average | 0.744 | | hard_first | 0.739 |
| fast | 0.725 | | easy_first | 0.737 |
| expert | 0.692 | | hesitant | 0.704 |
| slow | 0.666 | | erratic | 0.649 |
| novice | 0.604 | | tired | 0.637 |

There is **no measurable generalisation penalty** on partners the agent has
never met — including partner types varying along axes (strategy bias, 3-step
response latency, high speed jitter) that appear nowhere in training. The gap
is now −0.007, essentially zero, because reassignment lifted the worst
training partner (`novice`) from 0.420 to 0.604 and closed the spread that
previously made the gap look artificially negative.

### Zero-shot agent vs. a partner-specific agent

The abstract contrasts this work with "behavioral models trained using
historical interaction data". A **specialist** trained on `average` alone for
250k steps is exactly that. Both agents, 50 episodes per partner:

| Partner | Zero-shot | Specialist | Winner |
|---|---|---|---|
| average *(the specialist's own partner)* | **0.744** | 0.740 | zero-shot |
| fast | **0.725** | 0.724 | zero-shot |
| slow | 0.666 | **0.677** | specialist |
| expert | **0.692** | 0.688 | zero-shot |
| novice | **0.604** | 0.345 | zero-shot |
| tired | **0.637** | 0.575 | zero-shot |
| easy_first | **0.737** | 0.737 | tie |
| hard_first | **0.739** | 0.729 | zero-shot |
| hesitant | 0.704 | **0.720** | specialist |
| erratic | **0.649** | 0.595 | zero-shot |

The zero-shot agent wins on **8 of 10 partners, including the specialist's own
training partner**. The specialist collapses where partners diverge most from
what it saw: on `novice` it scores 0.345 and **finishes 0% of episodes**,
against the zero-shot agent's 100%.

That is the abstract's motivating claim, demonstrated rather than asserted:
training against one partner's history buys nothing on that partner and costs
a great deal on everyone else.

### Adaptive task allocation

The share of work the agent delegates tracks how capable the partner actually
is, and the held-out partners fall on the same curve as the seen ones:

| Partner | expert | fast | hard_first | average | easy_first | hesitant | slow | erratic | tired | novice |
|---|---|---|---|---|---|---|---|---|---|---|
| delegated | 50% | 48% | 44% | 41% | 38% | 29% | 21% | 19% | 18% | 9% |
| steps to finish | 25.1 | 26.1 | 26.8 | 27.1 | 28.4 | 31.6 | 35.4 | 37.9 | 38.9 | 42.3 |

Delegation falls monotonically with partner capability, and episode length
rises with it, with held-out partners interleaved throughout rather than
clustered at one end. The agent was never told which partner it had — this
comes entirely from the interaction signals in the abstraction layer.

Reassignment usage follows the same logic: **0.00 per episode with `expert`,
`fast` and `average`, rising to 1.00 with `novice`**. The agent only pulls work
back when the partner is actually struggling.

### Training

400,000 steps for the zero-shot agent, plus 250,000 for the specialist.
Both on CPU; the two ran concurrently, so wall-clock (53 min / 41 min) is
roughly double what either takes alone.

| | start | end |
|---|---|---|
| mean reward | +100.4 | +121.0 |
| CES | 0.588 | 0.686 |
| **episodes fully solved** | **83%** | **100%** |
| orders on time | 72% | 87% |
| synchronisation rate | 0.251 | 0.354 |
| policy entropy | 0.447 | 0.034 |
| value loss | 3.42 | 0.49 |
| **illegal-action rate** | **0.000** | **0.000** |

Ten charts in `results/`: `training_progress.png`, `coordination.png`,
`optimisation.png`, `zero_shot_gap.png`, `profile_ces.png`,
`policy_comparison.png`, `adaptive_allocation.png`,
`specialist_vs_general.png`, `ablation.png` and `ablation_by_partner.png`.

Two points of comparison:

* The original revision's 200,000-step run ended at −6.2 mean reward, 0.000 CES
  and 0% recorded completion.
* Before reassignment existed, the same agent reached 93.4% of episodes solved
  and scored 0.420 on `novice`. With it: **99.8% solved and 0.604 on `novice`**.
  That capability is worth more than any hyperparameter change made here.

### Is any of this significant?

Every policy is evaluated on the same episode seeds, so comparisons are paired.
`utils/stats.py` bootstraps a 95% interval on each mean CES and runs a paired
test of each difference against the learned agent, over all 500 episodes:

| Comparison | Difference in CES | 95% CI | p |
|---|---|---|---|
| PPO − random | +0.2456 | [+0.2351, +0.2558] | <0.001 |
| PPO − static | +0.2078 | [+0.1950, +0.2214] | <0.001 |
| PPO − specialist | +0.0367 | [+0.0289, +0.0449] | <0.001 |
| PPO − greedy | +0.0243 | [+0.0165, +0.0325] | <0.001 |

The margin over greedy is small but not noise: 500 paired episodes put the
interval well clear of zero.

---

## 2b. Component ablation

Five arms, identical 250,000-step budgets and seeds, one component changed
each, evaluated on identical episodes (`python ablation.py`).

| Arm | CES | 95% CI | vs `full` | p |
|---|---|---|---|---|
| curriculum | 0.697 | [0.692, 0.701] | +0.0153 | <0.001 |
| blind partner | 0.691 | [0.684, 0.697] | +0.0091 | <0.001 |
| no observer | 0.688 | [0.681, 0.695] | +0.0066 | <0.001 |
| full | 0.682 | [0.674, 0.689] | — | — |
| no reassignment | 0.679 | [0.669, 0.689] | −0.0023 | 0.61 |

**Curriculum sampling helps.** It is the only change that lifts every hard
partner at once — novice 0.598 → 0.625, tired 0.600 → 0.669 — and it reaches
100% solved and 92% on time. Worth adopting as the default.

**Reassignment looks neutral on average, but the mean hides it.** On `novice`
it is decisive: 0.598 with it, 0.428 without. It also flips the zero-shot gap
from −0.035 to +0.009. It earns its place on the hard partners and costs
nothing on the easy ones, which is exactly what selective use looks like.

**Removing the partner signals slightly improves the score.** This does not
support Novelty 5 as an efficiency claim, and it reproduces at both strengths:
blanking the smoothed summary gains +0.007, blanking the raw capability signals
too gains +0.009. The adaptive behaviour is untouched — a blind agent still
delegates 41.7% to an expert and 8.8% to a novice against the full agent's
41.7% / 8.3%.

The reason is that partner capability is already implicit in the scheduling
state: a slower worker stays `busy` for more steps, so the policy reads speed
from the timing of state transitions without needing a feature for it. The
explicit signals are redundant here, and the five extra dimensions cost a
little sample efficiency. The project still does what Novelty 5 describes —
generalised interaction signals instead of partner profiling — but this
environment does not demonstrate the dedicated feature block is *necessary*
for it. A partner whose capability could not be inferred from timing alone
(say, one whose error rate mattered more than their speed) would be the test
that separates the two.

**Caveat.** Each arm is a single training run at one seed, and the intervals
are over evaluation episodes rather than over seeds. They say how reliably
*these particular policies* differ, not how reliably the *components* do.
Several seeds per arm would be needed to claim the latter.

---

## 3. What the system does now

### 3.1 Environment

* **Concurrent orders.** Several six-task fulfilment orders are live at once,
  arriving at staggered steps, each with its own priority (0.2–1.0) and a
  deadline that tightens as priority rises. The agent must interleave orders
  rather than finish one before starting the next.
* **WAIT action.** The action space is `num_workers * num_tasks + 1`. Waiting
  while work is in progress is always legal and is not penalised.
* **Independent worker gating.** Each worker's availability is checked
  separately, so genuine parallel work is possible — and rewarded.
* **Conditional idle penalty.** A worker is only penalised for being idle when
  there is work it could actually take. Waiting while the team is blocked is
  free.
* **Environment-owned action mask.** Legality is computed once, by the
  environment, and published in `info["action_mask"]`. The agent reads it
  rather than re-deriving it, so policy and environment cannot disagree. A test
  verifies this by brute force over every action in every visited state.
* **Scales.** `--num_orders`, `--num_humans`, `--num_ai` are all parameters;
  observation and action dimensions are derived, and checkpoints record and
  enforce the shape they were trained with.

### 3.2 Human simulation

Five behavioural axes, none of them visible to the agent:

| Axis | Effect |
|---|---|
| speed | multiplier on task duration |
| error rate | per-step chance of an error that adds redo time |
| fatigue rate | accumulates while working; slows the worker up to 60% |
| **response latency** | steps between accepting a task and starting it — the "delayed response" behaviour |
| **speed jitter** | per-task random variation — the "erratic worker" behaviour |
| **strategy** | difficulty biases pace: `easy_first` is quick on easy tasks and slow on hard ones, `hard_first` the reverse |

The last three are new. `strategy` was previously stored and never read.

### 3.3 Zero-shot protocol

Training samples only from `average, fast, slow, expert, novice`. Held out
entirely: `tired, easy_first, hard_first, hesitant, erratic` — each differing
along an axis or combination the agent never trained on. A test asserts over
200 resets that no held-out profile leaks into training, and another asserts
that every held-out profile is behaviourally distinct from `average`.

### 3.4 Baselines

`evaluate.py --baselines` runs three scripted policies on identical seeds and
the same information:

| Policy | Behaviour |
|---|---|
| `static` | Predefined sequence, round-robin assignment, no adaptation — the traditional static workflow this project argues against |
| `greedy` | Strong heuristic: most urgent task first, to whichever free worker is estimated to finish soonest from observable speed alone |
| `random` | Uniform over legal actions |

None may take an illegal action, so no baseline is unfairly penalised.

### 3.5 Dashboard

`python app.py` → <http://localhost:5000>

Live SVG **task dependency graph** per order (nodes coloured by status and by
who is working, edges turning green as prerequisites clear, per-node progress
bars), order panels with deadline and slack, worker panels with human vitals, a
decision-by-decision event log, a **policy selector** (PPO vs each baseline on
the same scenario), a partner selector with held-out profiles labelled
"zero-shot", and a scenario selector for 1–3 orders and up to 2 AI + 2 humans.

### 3.6 Analytics

`python plot_training.py` writes eight charts to `results/`:
training progress, coordination behaviour, PPO optimisation diagnostics,
zero-shot gap over training, per-partner CES, policy comparison, and adaptive
task allocation.

---

## 4. How to reproduce

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

python tests/test_dts_zsc.py                                     # 25 tests
python train.py --total_steps 400000 --ent_coef 0.015            # ~25 min CPU
python evaluate.py --checkpoint checkpoints/agent_final.pt --baselines
python plot_training.py
python app.py
```

---

## 5. Mapping to the requirement document

| Requirement | Status | Where |
|---|---|---|
| RL-based dynamic task sequencing | done | `agent/ppo_agent.py`, `env/collaborative_env.py` |
| Simulated warehouse, picking / packing / dispatch | done | `env/task_definitions.py` |
| Zero-shot coordination with unseen partners | done | `human/simulated_human.py`, `utils/metrics.py::zsc_report` |
| **Novelty 1** Dynamic task sequencing model | done | order interleaving + WAIT + re-planning every step |
| **Novelty 2** Zero-shot human collaboration | done | 5 held-out partner types, gap reported |
| **Novelty 3** RL-driven coordination strategy | done | PPO with action masking, beats greedy and static |
| **Novelty 4** Adaptive task prioritisation | done | order urgency, deadlines, slack in the observation and reward |
| **Novelty 5** Human interaction abstraction layer | done | `agent/human_observer.py` |
| **Novelty 6** Scalable coordination framework | done | `env/workers.py`, N humans + N robots, tested at 3 orders / 4 workers |
| **Add-on 1** Interactive coordination dashboard | done | `app.py` |
| **Add-on 2** Task dependency graph visualisation | done | live SVG DAG per order in `app.py` |
| **Add-on 3** Human behaviour simulation | done | fast / slow / error-prone / **delayed** / **erratic** / strategy-biased |
| **Add-on 4** Performance analytics module | done | `utils/analytics.py`, 8 charts |
| Python · PyTorch · NumPy · Pandas · TQDM · Gymnasium | done | all six used; `tqdm` drives the training progress bar |

---

## 6. Every issue from the gap analysis, and its resolution

### Blocking problems

| # | Problem | Resolution |
|---|---|---|
| 3.1 | Agent never learned — flat reward, 0% completion | Root causes below fixed; reward now rises and converges (§2) |
| 3.2 | `_ai_current_task is None` gated *every* action, blocking parallelism; no WAIT action | Each worker gated independently; WAIT added and always legal; idle penalty made conditional. Test: `test_human_can_be_assigned_while_ai_is_busy`, `test_idle_penalty_does_not_fire_when_nothing_is_available` |
| 3.3 | `tasks_done` always 0 from an always-true expression | Environment maintains and reports `info["tasks_done"]`; `EpisodeTracker` reads it. Test: `test_completion_is_actually_counted` |
| 3.4 | `easy_first` identical to `average`; `strategy` never read | `strategy` implemented as a difficulty-dependent pace bias; 5 distinct held-out profiles. Tests: `test_held_out_profiles_are_behaviourally_distinct`, `test_strategy_bias_changes_the_difficulty_ordering` |
| 3.5 | Action mask contradicted the environment | Mask is now produced by the environment and read by the agent. Test: `test_mask_exactly_matches_environment_legality` (brute force) |
| 3.6 | Sync metric measured "has ever finished a task" | Uses `info["num_busy"]`, the live count of busy workers |

### Missing features

| Requirement | Resolution |
|---|---|
| Task dependency graph visualisation | Live SVG DAG per order in the dashboard |
| Performance analytics module | `utils/analytics.py`, eight charts, driven by `plot_training.py` |
| Adaptive task prioritisation | Order priority, deadlines and slack in the observation, reward and greedy heuristic |
| Scalable coordination framework | `env/workers.py` worker protocol; N humans + N robots |
| Order processing / multiple orders | Concurrent order queue with staggered arrivals |
| "Delayed responses" worker behaviour | `response_latency` — worker accepts a task but starts *n* steps later |

### Smaller defects

| Problem | Resolution |
|---|---|
| `plot_training.py` wrong path and column names | Rewritten as a CLI over `utils/analytics.py` |
| `requirements.txt` missing flask / matplotlib | Added, with versions; `tqdm` is now genuinely used |
| `app.py` divided by 200 while `MAX_EPISODE_STEPS` was 50 | Dashboard reads `env.max_steps` and reuses `CoordinationMetrics.ces` |
| Evaluation seed discarded by a second reset | `run_episode` resets exactly once. Test: `test_evaluation_is_reproducible` |
| Truncation treated as a true terminal | Bootstrapped: `r += gamma * V(s')` on time-limit truncation |
| Stale rollout-buffer slots trained on | Rollouts are always filled completely; buffer raises on overflow and resets after each update |
| No `__init__.py` | Added to `env/`, `agent/`, `human/`, `utils/` |
| No README | `README.md` — setup, MDP spec, protocol, metrics, structure |
| Zero commits | Repository committed |

---

## 6b. Two gaps found by checking the abstract against the code

After the work plan was complete, the project's own abstract and methodology
were re-read line by line against the implementation. Two claims were not
actually supported.

### Gap 1 — "prioritizes, defers or **reassigns** tasks"

The methodology promises three capabilities. Only two existed:

```
DEFER      -> supported      (WAIT action)
PRIORITISE -> supported      (order priority + deadline in observation and reward)
REASSIGN   -> NOT supported  (environment rejected it; mask marked it illegal)
PREEMPT    -> NOT possible   (no action returned an in-progress task to the pool)
```

Once a task started it ran to completion with that worker, whatever happened.

**Resolution.** An assignment action now either *starts* a pending task or
*reassigns* an in-progress one — the current holder is released via a new
`release()` method on both worker types, and the work already done is
forfeited. The cost is `-2.0 x (fraction of progress lost)`, so bailing out of
a bad assignment early is cheap and bailing out at 90% is expensive. Fatigue
and error history stay with the released worker, because they really did do
that effort and the coordinator should still be able to observe it.

`--no_reassignment` ablates the capability for comparison. The greedy baseline
gained a matching rescue rule so it remains a fair opponent, and it fires
rarely (about 0.6 times per episode) rather than thrashing.

Four new tests cover it, and the existing brute-force mask check now also
validates the expanded legality automatically.

**Effect.** This was not cosmetic. Retrained with the capability, the agent
went from 93.4% to **99.8% of episodes solved**, and the worst partner
(`novice`) from CES 0.420 to **0.604** with a 34% → 100% solve rate — because
it can now pull a critical-path task back off a struggling partner instead of
being stuck with the consequences of one bad assignment. It uses the capability
sparingly and sensibly: 0.00 reassignments per episode with capable partners,
1.00 with `novice`.

### Gap 2 — a baseline the abstract implies but did not exist

The abstract contrasts this work with two traditional approaches: "predefined
task sequences **or** behavioral models trained using historical interaction
data". Only the first had a baseline (`static`). The second — an agent trained
on one specific partner — is the more demanding comparison, and it is the one
that actually motivates zero-shot coordination.

**Resolution.** `train.py --train_profiles average` trains a specialist on a
single partner; `evaluate.py --compare specialist=<path>` runs it beside the
zero-shot agent across all ten partners and prints a per-partner winner column.
`train.py` refuses to train on a held-out profile, so a specialist can never be
built out of the zero-shot test set by accident.

**Result.** The zero-shot agent wins on 8 of 10 partners including the
specialist's own, and the specialist finishes **0%** of `novice` episodes
against the zero-shot agent's 100%. See §2 for the full table — this is now the
strongest single piece of evidence for the project's central argument, and it
was previously missing entirely.

### Additional work not in the original plan

* `utils/episode.py` — a single `EpisodeTracker` so training, evaluation and
  the dashboard assemble statistics identically.
* `utils/runner.py` — one shared episode loop, removing the duplicated and
  divergent loops in `train.py` and `evaluate.py`.
* `tests/test_dts_zsc.py` — 21 tests targeting the exact failure modes above.
* Reward scaling for the critic (rewards reported to the user stay unscaled),
  learning-rate annealing, and KL / clip-fraction diagnostics.
* Periodic zero-shot probes during training, so the generalisation gap can be
  plotted as a curve rather than a single end-of-run number.
* `.gitignore`, and the previous broken run archived to
  `checkpoints/legacy_broken_run/` rather than deleted.

---

## 7. Honest limitations

* **The abstraction layer is not shown to be necessary.** Blanking it, and even
  hiding the raw partner signals, slightly *improves* the score (§2b). The
  adaptive behaviour the project claims is real and measurable, but in this
  environment it comes from the scheduling state rather than from the dedicated
  partner features.
* **Every ablation arm is a single seed.** The intervals cover evaluation
  episodes, not training runs, so they understate how uncertain the
  *component-level* conclusions are.
* **The zero-shot gap is −0.007, not zero.** It is still marginally negative.
  Read it as "no measurable generalisation penalty", never as "generalises
  better than it trained" — with five profiles per group a difference of 0.007
  is inside the noise.
* **Partners are simulated, and we wrote the simulator.** There is circularity:
  the agent generalises across behaviours our own model produces. The distance
  between "handles our five held-out profiles" and "handles a real warehouse
  worker" is large and entirely unmeasured. No human-subject data is involved.
* **`greedy` is a strong baseline and beats us on one metric.** It delivers
  87.4% of orders on time against our 84.7%. The overall margin (+0.024 CES) is
  real but modest; the large margin (+0.208) is over the static workflow, which
  is the comparison the abstract actually makes.
* **One specialist, one partner.** The specialist comparison uses a single
  agent trained on `average`. A specialist per partner would be a more complete
  version of that experiment.
* **The AI worker is deterministic.** Only the human varies. A second
  stochastic robot would be a more demanding test of the coordination policy.
* **One scenario size was trained.** The framework scales, and a 3-order /
  4-worker configuration is tested for correctness, but every reported number
  is for 2 orders with 1 AI + 1 human.
* **Reassignment is free of human cost.** Taking a task off a person is modelled
  as a pure progress loss. A real worker would find repeated preemption
  demoralising, and nothing here captures that.
