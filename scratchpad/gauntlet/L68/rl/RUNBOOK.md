# RL on RoyaleSim -- runbook (L68)

Trainer: `pipeline/rl_royale.py`, defaults `pipeline/rl_royale.yaml` (every key commented with its source).
Design: `scratchpad/gauntlet/L68/rl_plan.md` (binding), `scratchpad/gauntlet/L67/e1_engine_rl_design.md` 3.2-3.6, 4.2.
All commands from the repo root; the trainer runs in the **Royale venv** (it has royalesim; the icebow venv does not).


## 0. Offline tests (icebow venv, no engine, ~1-2 min)
```
icebow/.venv/Scripts/python.exe -m unittest pipeline.tests.test_rl_royale -v
```

## 1. Smoke (~10-20 min on the busy box)
```
research/ext/Royale/.venv/Scripts/python.exe -m pipeline.rl_royale --config pipeline/rl_royale.yaml --run smoke_<date> --smoke
```
E=4, G=2, 1 actor, 8 in flight, 2 updates, screen on 8 held-out entries, pro agreement on 1,000 VAL rows. It checks:
u0000 reloads through `engine_play.load_model` with identical pro agreement; update 0 is on-policy (max |ratio-1| < 1e-3,
KL < 1e-5 per head, else AssertionError = a bug); a fresh learner through `--resume` restores update / beta / optimizer
/ rng / guards. Last line `SMOKE PASS` (exit 0) or `SMOKE FAIL: <check>` (exit 1). Delete its two dirs afterwards
(`scratchpad/gauntlet/L68/rl/smoke_<date>`, `icebow/data/bench/rl_royale/smoke_<date>`).

## 2. Start
```
research/ext/Royale/.venv/Scripts/python.exe -m pipeline.rl_royale --config pipeline/rl_royale.yaml --run <name>
research/ext/Royale/.venv/Scripts/python.exe -m pipeline.rl_royale --config pipeline/rl_royale.yaml --run <name> E=8 n_actors=2     # any key=value override
```
Refuses an existing non-empty run dir (use a new name, or `--resume`). Startup (~5-10 min): loadable-entry scan
(cached in `entries.json`: train 299, held-out 58), init pro agreement on all 13,761 v3 VAL rows (`init_proagree.json`),
init held-out screen on the 58 (`init_screen.json`), `<name>_u0000.pt` (= the init). Then one update per ~64 matches.

Files: `scratchpad/gauntlet/L68/rl/<name>/` = `train.log` (one human line per update), `train_log.jsonl` (everything),
`config.yaml`, `pid.json` (learner + actor PIDs while running), `entries.json`, `init_*.json`.
Checkpoints: `icebow/data/bench/rl_royale/<name>/` = `<name>_latest.pt` (every update), `<name>_u{NNNN}.pt`
(NNNN = updates done, every `save_every`), `<name>_crash_*.pt` on a crash.

## 3. Watch -- the fields that matter (`train_log.jsonl`, type "update")
| field | healthy | why |
|---|---|---|
| `plays_per_min` vs update 0 | within 0.6x-1.6x (stop rule, 2 in a row) | gate collapse / over-play (engA) |
| `kl_cell`, `beta_next` | kl_cell near `kl_target` 0.10; beta moving inside [0.03, 3] | the leash; beta at 3.0 with KL_cell > 0.5 = stop |
| `kl_gate`, `kl_card` | small (< 0.1) | gate drift is the historical collapse channel |
| `mixed_group_share` | > 0.3 | share of entries whose G rollouts disagree = the gradient's real sample size |
| `entropy` vs `entropy_init` | same order | a head going deterministic |
| `clip_frac`, `ratio_mean` | clip_frac < ~0.2, ratio_mean ~1 | step size |
| `first_minibatch.ratio_maxdev` | < 1e-3 every update | on-policy integrity (asserted at update 0) |
| `screen.delta_pp`, `ci_lo_pp`/`ci_hi_pp`, `better`/`worse` | the signal; stop = delta <= -10 pp AND CI upper < 0 | held-out RoyaleSim, 58 entries x seeds 0,1,2, entry-clustered paired vs init (every `screen_every`) |
| `proagree_delta_pp` | > -1 pp cell | tripwire: cell < -1 pp AND screen delta <= 0 = stop; hard stop -3 pp cell/card, -0.05 gate |
| `outlived_win_share`, `low_delivered_win_share`, `ghost_refused_per_match` | near their update-0 values; refused EMA below `guards.ghost_refused_limit` = max(2x, +1.0) of update 0 (logged at update 0) | ghost-exploit guards (E1 4.2); `ghost_undelivered_per_match` is a monitor only |
| `winrate` | descriptive only | train entries are revisited; never a verdict |
| `visits_distinct`, `visits_max` | -- | pool revisits (299 entries) |
| `wall_*`, `actor_s_per_match`, `*_gpu_peak_mb` | -- | throughput |

**Screen noise (measured L68, run noise_L68, 3 updates at the default config, KL_cell 0.002-0.006):** paired deltas
-0.6 / +1.1 / +0.6 pp, 95% CI half-width ~6-8 pp (e.g. [-8.0, +7.5]), and 35-46 of the 174 (tag, k) pairs flip
(better/worse ~22/23) even though the policy has barely moved -- the greedy live rule turns tiny weight changes into
whole-match flips. So a screen delta inside roughly +-8 pp says nothing; read the CI, not the point estimate. (The old
8-entry, 1-seed smoke screen swung -25 pp on 2 flips.)

**Gradient clip (measured):** every step's gradient norm exceeds `grad_clip` 0.5 (mean 1.3-2.0), so each step is a
fixed-length step in the gradient's direction; at lr 1e-5 KL_cell still stays ~0.002-0.006 per update. Because KL_cell
is far below kl_target / 1.5, adaptive beta halves every update (0.3 -> 0.15 -> 0.075 -> 0.0375) and sits at its 0.03
floor from ~update 4. That is the rule working as written, not a fault. Watch: KL_cell climbing toward 0.10 over tens
of updates (then beta starts rising again); `clip_frac` above ~0.2 or `ratio_mean` drifting off 1 (steps too large for
the clip); if KL_cell is still < 0.01 after ~20 updates, the policy is effectively frozen at this lr -- raising lr is an
owner decision, like kl_target.

`tail -f` the human log: `Get-Content scratchpad/gauntlet/L68/rl/<name>/train.log -Wait -Tail 20`.

## 4. Stop
Create `scratchpad/gauntlet/L68/rl/<name>/STOP` (any content). The learner finishes the current update, saves
`_latest.pt`, logs `STOP after update N: STOP file present`, closes the actors, exits 0. Every automatic stop rule ends
the same way with its reason on the STOP line (a non-finite loss/parameter writes a crash save instead and leaves
`_latest.pt` at the last good update). Do not kill the learner mid-update unless it hangs; if you must, kill the learner
PID from `pid.json` and then the actor PIDs (actors also exit on their own within ~30 s once the learner is gone).
Verify with `Get-CimInstance Win32_Process -Filter "name='python.exe'" | ? CommandLine -match rl_royale`.

## 5. Resume
Delete the STOP file first (resume refuses while it exists), then
```
research/ext/Royale/.venv/Scripts/python.exe -m pipeline.rl_royale --config pipeline/rl_royale.yaml --run <name> --resume [key=value ...]
```
Continues from `<name>_latest.pt`: update counter, beta, Adam state, entry-sampling rng, visit counts, stop-rule
counters/EMAs, init baselines (pro agreement, init screen per tag). A changed config is allowed and logged
(`RESUME with a changed config`), e.g. a raised `max_updates`.

## 6. Gate a checkpoint
RoyaleSim held-out, 3 seeds (paired vs the init):
```
icebow/.venv/Scripts/python.exe -m pipeline.rl_gate --commands --engine royale ^
    --init-ckpt icebow/data/pipeline/s1_icebow_v6aug_s1.pt --cand-ckpt icebow/data/bench/rl_royale/<name>/<name>_u0050.pt ^
    --out-root scratchpad/gauntlet/L68/rl/<name>/gate_royale_u0050
```
run the two printed `e1_eval` lines, then
```
icebow/.venv/Scripts/python.exe -m pipeline.rl_gate --init scratchpad/gauntlet/L68/rl/<name>/gate_royale_u0050/init ^
    --cand scratchpad/gauntlet/L68/rl/<name>/gate_royale_u0050/cand --json scratchpad/gauntlet/L68/rl/<name>/gate_royale_u0050/report.json
```
Real engine (the verdict that counts): same with `--engine real` (boot both engine slots first, ports 38031/38032; see
HANDOFF / `e1/_boot.ps1`), four printed lines (init/cand x slot0/slot1). Pro agreement for criterion (ii) is not in
rl_gate: read `proagree` from train_log.jsonl or run
`icebow/.venv/Scripts/python.exe -m pipeline.eval_s1 icebow --data icebow/data/pipeline/s1_dataset.npz <ckpt>`.
No "best on held-out" selection: gate the checkpoint you decided to gate before looking at screens.

## 7. The KL budget is the owner's
`kl_target` (default 0.10) is NEVER changed by the trainer. Raising it is a manual owner decision, taken only after a
real-engine `rl_gate` PASS on a checkpoint trained at the current target; then `--resume` with `kl_target=<new>` on the command
line (logged as a changed config), and record the decision in HANDOFF.
