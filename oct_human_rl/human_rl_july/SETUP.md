# Setup and run

Everything below takes about five minutes. You need **Python 3.10 or newer**
and an internet connection for the one-time install.

Check your Python:

```bash
python3 --version
```

---

## 1. Create a virtual environment

A virtual environment keeps this project's packages separate from the rest of
your system. Run these from inside the unzipped project folder.

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Your prompt now starts with `(.venv)`. Everything after this runs inside it.

> **Ubuntu/Debian:** if `python3 -m venv` reports that the `python3-venv`
> package is missing, install it once with
> `sudo apt install python3-venv python3-pip` and run the command again.

---

## 2. Install the dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Save several gigabytes:** the default PyTorch wheel bundles CUDA/GPU
> support that this project never uses. Nothing here needs a GPU. To pull the
> much smaller CPU build instead, install torch first:
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

---

## 3. Confirm the install works

```bash
python -m pytest tests/test_dts_zsc.py -q
```

All 43 tests should pass in roughly 15 seconds. If they do, the environment,
the agent and the dashboard are all working.

---

## 4. Start the dashboard

```bash
python app.py
```

Then open **http://localhost:5000** in a browser.

Press `Ctrl+C` in the terminal to stop the server.

### What to click first

1. **Workload** — choose anything from 1 to 6 orders.
2. **Team** — up to 2 AI and 2 human workers.
3. **Order priority** — pick **Custom sequence** to rank the orders yourself,
   then click the orders in the sequence you want them worked, for example
   1, 2, 3, 6, 5, 4. Rank #1 becomes the most urgent and gets the tightest
   deadline. Orders still arrive on their own schedule, so a ranking cannot
   start an order before it exists.
4. **Run** — watch the agent assign work; **Step** advances one decision at a
   time if you want to read each one.

The trained PPO checkpoint covers the default scenario (2 orders, 1 AI + 1
human). Any larger scenario falls back to the Greedy coordinator and says so
in a banner — no untrained model is ever presented as PPO. To train a policy
for a larger scenario, see section 9 of `README.md`.

---

## 5. Optional: reproduce the reported numbers

```bash
python evaluate.py --baselines     # zero-shot evaluation + baseline comparison
python ablation.py                 # component ablation study
python train.py                    # retrain from scratch (slow)
```

Results are written to `results/` and checkpoints to `checkpoints/`.

Keep `--baselines` on the evaluation command: `results/evaluation.json` feeds the
dashboard's **Performance comparisons** panel, and a run without it overwrites
the file without the baseline numbers that panel needs. To get the saved
results back, run `git checkout -- results/`.

---

## 6. Optional: PPO for every workload and team

The default PPO model only fits 2 orders with 1 AI + 1 human, because a PPO
network has a fixed number of inputs and outputs. For any other scenario the
dashboard runs the Greedy coordinator and says so in a banner. To train a PPO
model for every other dashboard scenario (17 of them; this takes hours on a CPU):

```bash
python scripts/train_scenarios.py --list     # what is trained, what is missing
python scripts/train_scenarios.py            # train everything that is missing
python scripts/train_scenarios.py --only 3o_2ai_2h   # or just one scenario
```

Models go to `checkpoints/scenarios/<orders>o_<ai>ai_<humans>h/`, and the
dashboard uses them automatically after a restart. Each one is compared with Greedy
when it finishes, and the banner shows that comparison. The default model and
the reported research results are never changed.

---

## Where things are

| Path | What it holds |
|------|---------------|
| `app.py` | the interactive dashboard |
| `env/` | the warehouse environment, tasks and workers |
| `agent/` | the PPO agent and the human behaviour observer |
| `human/` | the simulated human partner profiles |
| `baselines.py` | the scripted policies the agent is compared against |
| `tests/` | the test suite |
| `docs/` | walkthroughs, the UI guide and the handoff guide |
| `results/` | evaluation output and figures |
| `README.md` | the full technical description |

---

## If something goes wrong

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'flask'` | The virtual environment is not active. Re-run the activate command from step 1. |
| `python3 -m venv` fails on Ubuntu | `sudo apt install python3-venv python3-pip` |
| Port 5000 is already in use | Stop the other program, or edit the `app.run(...)` line at the bottom of `app.py` to use another port. |
| The install downloads gigabytes | Use the CPU-only PyTorch command in step 2. |
