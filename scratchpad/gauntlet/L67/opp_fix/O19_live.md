# O19 -- LIVE wiring for the opponent-elixir V2 estimate (one-line-config-change prep)

Repo HEAD at start: 3d8ad52. Python: icebow/.venv/Scripts/python.exe. No engine/live run. No commits.

## Verified before editing

- `diff icebow/src/clashrl/play.py hogeq/src/clashrl/play.py` -> 385 lines differ. NOT byte-identical.
  Per ticket rule: **editing icebow only**; hogeq/src/clashrl/play.py is untouched.
- Live construct site: `icebow/src/clashrl/play.py:400` `_opp_elx = OpponentElixirEstimator(_db)`.
- `update()` call site: `:558` `_est = _opp_elx.update(float(my_elixir), dets, now)`, where `dets` (`:547`)
  is already whitelist-filtered (`d.team == "enemy" and d.base in detector_cards`).
- `record_my_play`: `:1130` `_opp_elx.record_my_play(...)`. `reset`: `:1196` `_opp_elx.reset(...)`.
- `_student_opp_elixir` flag: `:186`, gates `_oe` at `:884` -- confirmed a SEPARATE switch from the new one.
- TeamTracker instance already exists in play.py for its own tagging: `_team_tracker = TeamTracker(...)`
  constructed `:422-441`, `.tag()` called synchronously at `:518` (non-perception-thread path) and by
  `PerceptionLoop._tracker.tag(...)` (icebow/src/clashrl/perception.py:200) on the SAME instance when the
  perception thread is running (`_ploop = PerceptionLoop(cfg, _detector, _team_tracker, ...)` at `:496`).
  So `_team_tracker._tracks` is current by the time `_threat_extra()` reaches the estimator update, under
  BOTH perception modes, with no second tracker needed. Confirms item 6 of the ticket does NOT apply
  (a reusable tracker exists on this exact code path).
- Billing pattern mirrored from `pipeline/opp_est_audit.py` (`make_team_tracker` :892-914, `tt_bill_dets`
  :941-994) READ, NOT imported (pipeline/ never imported from the live path): stamp a per-tracker monotonic
  uid into each track dict on first sight (`tr["_bill_uid"]`), bill a track the first time it is
  `team == "enemy"` and `hits >= tracker.min_hits`, key the "already billed" set on the uid (not `id(dict)`,
  which the audit module's own O16 FIX 1 measured as unsafe under CPython address recycling).
- `TeamTracker` track dict fields confirmed in `icebow/src/clashrl/replay_mine.py`: `tr["team"]`,
  `tr["hits"]` (:452-453), `tr["base"]`, `tr["x"]`, `tr["y"]` (:436-452), `self.min_hits` (:311).
- Existing test home for play.py wiring: `icebow/tests/test_play_tracker_wiring.py` (source-parses play.py's
  `_team_tracker = TeamTracker(...)` construction, then exercises a reconstructed tracker for behaviour).
  No test file already covers `_opp_elx`'s wiring specifically (`test_opponent_elixir.py` /
  `test_opponent_elixir_v2.py` test the estimator CLASSES in isolation, not play.py's use of them).
  Added the new tests to `test_play_tracker_wiring.py`.

## Design (minimal diff)

- New code-default-OFF flag `play.opp_elixir_v2`, read exactly like `_student_opp_elixir` (`:186`):
  `bool(cfg.get("play", "opp_elixir_v2", default=False))`.
- `_opp_elx` construction (`:400`) becomes `OpponentElixirEstimatorV2(_db) if _opp_elixir_v2 else
  OpponentElixirEstimator(_db)` -- OFF path is the exact same call as before.
- New MODULE-LEVEL helpers (not nested inside `play()`, so they are directly importable/testable):
  `_BilledDet` (a `.base/.cx/.gy/.team` shim) and `_opp_elixir_v2_bill(tracker, bill_state)`, which mirrors
  `tt_bill_dets` but never calls `tracker.tag()` itself (play.py's own tag() call, sync or via the
  perception thread, already ran this tick for every other consumer of `_team_tracker`).
- `:558` update call becomes an if/else: OFF branch is byte-identical to today
  (`_opp_elx.update(float(my_elixir), dets, now)`); ON branch bills via `_opp_elixir_v2_bill(_team_tracker,
  _opp_bill)`, filters the billed dets through the SAME `detector_cards` whitelist (applied to the
  tracker's output, mirroring `filter_whitelisted_billed_dets`'s placement), then calls `.update()` with
  those.
- `_opp_bill = {"uid_counter": itertools.count(1), "billed": set()}` created once near `_opp_elx`
  construction (`:400`); `_opp_bill["billed"].clear()` added at the match-reset block (`:1196`) alongside
  `_opp_elx.reset(...)` -- the uid counter itself is never reset (matches `tt_bill_dets`'s own "kept one per
  (match, tracker)" billed-set semantics; a fresh match's tracks are fresh dicts either way).
- `_student_opp_elixir` (`:186`, `:884`) is UNTOUCHED -- stays its own independent switch.

## Edits made

- `icebow/src/clashrl/play.py`: `import itertools`; module-level `_BilledDet` + `_opp_elixir_v2_bill(tracker,
  bill_state)` (mirrors `tt_bill_dets`, does not call `.tag()` itself); `_opp_elixir_v2` flag read next to
  `_student_opp_elixir`; `_opp_elx` construction ternary; `_opp_bill` state dict; the `update()` call site
  if/else (OFF branch line is byte-for-byte what it was before this ticket); `_opp_bill["billed"].clear()`
  next to `_opp_elx.reset(...)` at the match-reset block. `record_my_play` (`:1130+diff`) and
  `_student_opp_elixir`/`_oe` (`:186`, `:884+diff`) are UNTOUCHED. `git diff --stat`:
  `icebow/src/clashrl/play.py | 74 ++++++++++++++++++++++++++++++++++++++++++++--` (71 insertions, 3
  deletions -- the 3 deletions are the single `_opp_elx = OpponentElixirEstimator(_db)` and
  `_est = _opp_elx.update(...)` lines each becoming multi-line, plus the reset-block line moving).
- `icebow/tests/test_play_tracker_wiring.py` (the existing home for play.py-wiring tests -- no test file
  already covered `_opp_elx`'s wiring; `test_opponent_elixir[_v2].py` test the estimator CLASSES only):
  added `import itertools`, `OpponentElixirEstimator`/`OpponentElixirEstimatorV2`/`clashrl.play` imports,
  and two new `TestCase` classes, `OppElixirV2Wiring` (source-parsed, same convention as `PlayTrackerWiring`)
  and `OppElixirV2Billing` (behavioural, against `play._opp_elixir_v2_bill`/`play._BilledDet` + a real
  `TeamTracker`) covering all six ticket items:
  (i) OFF path construction+update-call source-pinned byte-identical to before.
  (ii) `test_a_one_frame_phantom_is_never_billed` / `test_a_two_frame_unit_is_billed_exactly_once...`.
  (iii) `test_an_expired_then_recreated_track_is_billed_again`.
  (iv) `test_on_path_bills_via_the_shared_tracker_and_reapplies_the_whitelist` (source) +
  `test_the_whitelist_still_filters_billed_dets` (behavioural).
  (v) `test_record_my_play_is_unconditional_in_both_modes`.
  (vi) `test_student_opp_elixir_is_untouched_and_independent_of_the_new_flag`.
  hogeq/tests/test_play_tracker_wiring.py (currently byte-identical to icebow's pre-edit copy) is left
  UNCHANGED -- out of the write set; it still tests hogeq's own untouched play.py and is unaffected.

## Verification run

`cd icebow && PYTHONPATH=src ../icebow/.venv/Scripts/python.exe -m pytest tests/test_opponent_elixir.py
tests/test_opponent_elixir_v2.py tests/test_play_no_cnn.py tests/test_play_tracker_wiring.py
tests/test_live_opponent.py -q` -> **92 passed** (17 of those in `test_play_tracker_wiring.py`, 12 new).
`ast.parse` on the edited `play.py` -> syntax OK. `git diff --stat` limited to `icebow/src/clashrl/play.py`
and `icebow/tests/test_play_tracker_wiring.py`; `hogeq/**` and `pipeline/**` show no changes from this
session (pipeline's own working-tree diff, from the concurrent O18 worker, predates and is untouched by
this session).

## Exactly what changes on the live path, per flag combination

- `opp_elixir_v2=False, student_opp_elixir=False` (today's default): UNCHANGED. V1
  `OpponentElixirEstimator`, fed every raw whitelisted enemy det every frame; S1 never sees the estimate
  (`_oe=None`).
- `opp_elixir_v2=False, student_opp_elixir=True`: UNCHANGED from before this ticket (L67f/L67g behaviour).
  V1 as above; S1 receives `_opp_elx._est` (elixir units).
- `opp_elixir_v2=True, student_opp_elixir=False`: V2 `OpponentElixirEstimatorV2` replaces V1; its input is
  no longer every raw det but one `_BilledDet` per `_team_tracker` track the FIRST time it is confirmed
  (`hits >= _team_tracker.min_hits`, verdict `enemy`), whitelist-filtered the same way; `record_my_play`/
  `reset` unaffected; S1 still sees `_oe=None` (independent switch).
- `opp_elixir_v2=True, student_opp_elixir=True`: V2 + tracker-billing as above, AND S1 now receives THIS
  estimator's `_est` instead of V1's.
- In every combination, `_team_tracker` itself, its `.tag()` call site, the aim assists, threat/identity
  vectors and everything else reading `dets`/`dets_all` are untouched -- only `_opp_elx`'s construction and
  its `update()` input change.

## Exact revert

`git checkout -- icebow/src/clashrl/play.py icebow/tests/test_play_tracker_wiring.py` (repo HEAD 3d8ad52,
before this session's edits) -- or restore byte-for-byte from
`scratchpad/gauntlet/L67/opp_fix/backup/play.py.o19.orig` and
`scratchpad/gauntlet/L67/opp_fix/backup/test_play_tracker_wiring.py.o19.orig` (both verified `md5sum`-equal
to the pre-edit HEAD files before editing began). No config default was touched, so simply not setting
`play.opp_elixir_v2` in config.yaml is already a full behavioural revert without touching code at all.

## Attempt 2

Blind verifier returned PASS_WITH_NOTES on attempt 1 with one MEDIUM defect (FIX 1) and a design gap
(FIX 2), plus two LOW items (FIX 3 tests, FIX 4 docs). Coordinator relayed all four; addressed in this
order. Repo HEAD moved from 3d8ad52 to `ab76edb` between attempts (the concurrent O18 worker's own commit,
`pipeline/opp_est_audit.py` + `pipeline/tests/test_opp_est_audit.py` + this progress file, which its commit
picked up as an untracked file) -- **verified `play.py`/`perception.py` are byte-identical between the two
commits** (`git show 3d8ad52:...|md5sum` == `git show ab76edb:...|md5sum` for both files), so nothing about
this ticket's baseline changed.

**Restore step**: `cp scratchpad/gauntlet/L67/opp_fix/play_o19_wip.py icebow/src/clashrl/play.py` and the
test-file equivalent, restoring attempt 1's edits (the lead had reverted the working tree to HEAD so the
owner could run live on verified code). `git diff --stat` after the restore matched attempt 1's own
diffstat exactly (74/137 insertions) before any attempt-2 changes were made, and the 19 attempt-1 tests
still passed -- confirmed a clean restore before touching anything.

**Backup-verification note**: `git show HEAD:<file> | md5sum` does NOT match this session's `.o19.orig`
backups directly -- this repo's blobs are stored LF-only while Windows checkouts here are CRLF (confirmed
by git's own "LF will be replaced by CRLF" warnings on every commit-adjacent git call). Stripping `\r`
(`tr -d '\r'`) from both sides makes every backup match its HEAD blob exactly; `git status --porcelain`
being clean at the moment each backup was taken (the actually-authoritative check, since it is CRLF-aware)
was already confirmed for `play.py`/`test_play_tracker_wiring.py` at the start of attempt 1 and for
`perception.py` at the start of attempt 2, before any of them were first read or copied.

### FIX 1 (MEDIUM, verifier) -- perception-thread race on `_team_tracker`'s track dicts

**The defect, confirmed by reading**: `TeamTracker.tag()` (`replay_mine.py`) does
`trk["x"], trk["y"], trk["t"] = dx, dy, t` on an EXISTING track dict (not a fresh one), and
`PerceptionLoop._run()` calls `self._tracker.tag(dets, now)` on the SAME `_team_tracker` instance play.py
hands it, inside `with self._lock:`. Attempt 1's `_opp_elixir_v2_bill` read `tr["x"]`/`tr["y"]` from that
same tracker with NO lock at all when `_ploop` was running -- exactly the every-other-guarded-access
pattern play.py already follows for `record_play`/`set_towers`/`snapshot`/`enemy_tracks`, which attempt 1
missed for this one new read path.

**Fix**: moved `_BilledDet`/`_opp_elixir_v2_bill` from play.py into `icebow/src/clashrl/perception.py` (a
lightweight, stdlib-only module play.py already imports `PerceptionLoop` from -- no import cycle), and
added `PerceptionLoop.bill_confirmed(bill_state, whitelist=None)`, a lock-guarded passthrough matching
`record_play`/`set_towers`/`snapshot`/`enemy_tracks`. It runs `_opp_elixir_v2_bill` (the WHOLE scan) inside
`with self._lock:`, copying every field it needs into fresh, immutable `_BilledDet` objects before
releasing -- so nothing returned can be torn, and the lock is released BEFORE the caller's `_opp_elx
.update()` runs (never held across the estimator call). Whitelist filtering (plain string containment on
the already-copied dets) runs after release. play.py's call site now branches on the SAME
`if _ploop is not None and _ploop.running:` pattern used everywhere else in the file: locked
`_ploop.bill_confirmed(...)` when perception owns the tracker, or the direct unlocked call when this act
loop is the tracker's only reader/writer (no perception thread running -- no race possible by
construction).

**Backup**: `icebow/src/clashrl/perception.py` backed up to
`scratchpad/gauntlet/L67/opp_fix/backup/perception.py.o19.orig` BEFORE editing (md5 matches HEAD after
CRLF-stripping, see above) -- perception.py was not in ticket O19's original WRITE SET, but FIX 1 explicitly
instructs adding a locked passthrough there, so it was necessarily added to the write set for this ticket;
called out here rather than silently expanded.

**Tests** (new, `PerceptionBillConfirmedLockTests` in test_play_tracker_wiring.py): the scan runs while
`self._lock.locked()` is True (proven by monkeypatching `_opp_elixir_v2_bill` to record the lock state when
called), the lock is released before `bill_confirmed` returns, `bill_confirmed` agrees with the unlocked
helper on an identical tracker, a 1-hit track is still not billed through the passthrough, and the
whitelist is applied after the locked scan. A genuine two-thread race is not asserted directly (it would be
flaky by nature); the lock-state monkeypatch test instead proves the STRUCTURAL guarantee (the scan cannot
run while the lock is free) that makes the race impossible by construction, matching how `_run()`'s own
`tag()` call is documented and guarded.

### FIX 2 (design change) -- `play.opp_elixir_v2_shadow`, observe-only

**Confirmed the verifier's finding by reading**: `mem[5]` is written from `_est` unconditionally
(`play.py`, inside `_threat_extra`), `env.opp_mem_slot5` defaults to `"opp_estimate"` in `config.yaml`, and
the lead confirmed `icebow/data/policy_rl.pt` exists (`_no_cnn` is False, so the CNN's `PolicyNet` forward
pass consumes `threat_vec`, which `mem` folds into). So `play.opp_elixir_v2` alone was never
observe-only -- flipping it changes what the CNN (and, if `student_opp_elixir`, S1) actually decide on.
Attempt 1 never claimed otherwise, but the ticket's premise ("prepare... so enabling it later is a
one-line change" without a safe comparison mode first) needed this addition.

**Built**: `play.opp_elixir_v2_shadow` (code default False, read the same way as the other two flags).
`_opp_elx_shadow` is constructed ONLY when this flag is True, and is ALWAYS the class OTHER than whichever
`_opp_elx` is (`OpponentElixirEstimator` if `opp_elixir_v2` is True, else `OpponentElixirEstimatorV2`) --
so the shadow log is always a genuine V1-vs-V2 comparison, never V2-vs-itself when both flags happen to be
on. Both estimators are fed every tick the shadow flag is on: whichever one is NOT driving `mem[5]` gets
fed the SAME input its non-shadow counterpart would use (raw whitelisted `dets` for a V1 shadow, tracker-
billed dets for a V2 shadow) -- one billing scan (`_opp_elixir_v2_bill`/`bill_confirmed`, shared whenever
EITHER `opp_elixir_v2` or `opp_elixir_v2_shadow` is on) feeds both, never a second independent tracker
walk. Logged at the EXISTING per-frame `[play]` cadence (no new sink):
`[play] opp_elixir_v2_shadow V1=.. V2=.. diff=.. driving=V1|V2 wall=..`. `mem[5]` and the S1 `_oe` value
are read from `_opp_elx`/`_est` ONLY -- `_opp_elx_shadow`/`_shadow_est` never appear on either of those two
lines (asserted by test, see below). `_opp_elx_shadow.reset()` runs alongside `_opp_elx.reset()` at the
match boundary, guarded on `is not None`.

**CPU cost, measured offline** (synthetic in-process microbenchmark, `icebow/.venv/Scripts/python.exe`, an
8-confirmed-track `TeamTracker`, `pytest`/live capture NOT involved -- see the exact script in this
session's transcript, not saved to the repo since it is throwaway):
- The lock-held billing scan alone (`_opp_elixir_v2_bill`, what runs inside `PerceptionLoop.bill_confirmed`
  's lock): **~1.4 microseconds/call** on 8 live tracks.
- Shadow mode's full added cost per frame (billing scan + a second estimator's `.update()` call, V1's own
  `.update()` costs ~0.3 microseconds/call by comparison): **~2.0 microseconds/frame**.
This is a synthetic microbenchmark (no frame capture, no detector inference, no real perception-thread
contention) -- it measures the estimator/tracker-scan CPU cost in isolation, not any GIL/lock-contention
effect from a real ~10 Hz perception thread also holding the same lock; that combined effect is UNMEASURED
(would need an actual live or engine session, which this ticket must not run). Given the isolated cost is
~4 orders of magnitude below both the ~100 ms perception period and the 1.5 s default `act_period`, it is
not expected to be observable, but that expectation is stated as such, not measured end-to-end.

**Tests** (new, `OppElixirV2ShadowWiring`): flag defaults off (behavioural: `eval()` of the real
`_opp_elixir_v2_shadow = ...` expression against a real loaded `Config`); shadow construction is always the
class OTHER than `_opp_elx`, for both values of `opp_elixir_v2` (behavioural eval, and `None` when the
shadow flag itself is off); the `mem[5]` and `_oe` lines never mention the shadow estimator; the shadow
reset sits with the primary reset (text/ordering pin, documented why); the billing input is shared between
the two flags (text pin -- the shared `if _opp_elixir_v2 or _opp_elixir_v2_shadow:` guard).

### FIX 3 (LOW, tests)

- `test_v1_and_v2_estimators_both_accept_the_billed_det_shape` (a bare `0 <= norm <= 1` check, which both
  classes satisfy UNCONDITIONALLY via their own saturation clamp -- a `_BilledDet` missing `.base`/`.cx`
  would still pass) -> replaced with
  `test_v1_and_v2_estimators_charge_the_billed_dets_own_known_cost`, which asserts the EXACT charge: one
  confirmed `"giant"` det must move `est._opp_spent` by exactly `db.elixir("giant")` and `est._est` by
  exactly `5.0 - cost`, for BOTH classes (giant has no SPAWNS/BODIES entry and no sibling track, so none of
  V2's extra rules change this from V1's plain accounting -- a genuine agreement check, not just "some
  number in range").
- `test_record_my_play_is_unconditional_in_both_modes` (grepped 200 raw characters of source before the
  call site for the absence of `"if _opp_elixir_v2"`) -> replaced with an AST-based check
  (`_RecordMyPlayGuardVisitor`): walks play.py's real parse tree tracking the stack of enclosing `if` TESTS,
  finds the actual (sole) `_opp_elx.record_my_play(...)` call node, and asserts none of its enclosing `if`
  tests mention `_opp_elixir_v2` -- a structural claim about control flow, immune to comment text or
  reformatting around the call site (unlike a fixed-character-window grep).
- The other four `OppElixirV2Wiring` tests: converted two to genuine behaviour by extracting the REAL
  right-hand-side expression from source (a generalised `_rhs_of(name)` helper, same
  extract-then-reconstruct idea as this file's pre-existing `_construction()` for `TeamTracker`) and
  `eval()`-ing it against a controlled namespace: `test_off_path_construction_actually_builds_v1_not_v2` /
  `test_on_path_construction_actually_builds_v2` (real `isinstance` checks against the real classes, for
  both flag values) and `test_the_flag_reads_like_student_opp_elixir_and_really_defaults_to_False`
  (`eval()`s the real `cfg.get(...)` call against a real loaded `Config`, proving config.yaml really leaves
  it unset). `test_student_opp_elixir_is_untouched_and_independent_of_the_new_flag` converted similarly, but
  more pointedly: it `eval()`s the real `_oe = ...` expression in a namespace that DELIBERATELY OMITS
  `_opp_elixir_v2` -- if the code ever came to depend on that name, this raises `NameError` instead of
  silently passing a substring check. Two of the six stayed TEXT PINS, each labelled and justified in the
  test file itself: `test_off_path_update_call_is_unchanged` (no free-standing local variable exists to
  `eval()` the OFF branch's `update()` call against outside `play()` itself) and
  `test_match_reset_clears_the_billed_set_alongside_the_estimator_reset` (a source-ORDERING claim between
  two reset lines, only meaningful in the real per-match reset control flow). The seventh,
  `test_on_path_bills_via_the_shared_tracker_and_reapplies_the_whitelist`, is also left a text/structural
  pin (checks the exact call text plus, via a bounded slice, that no second `TeamTracker(` construction
  appears on the billing path) for the same reason -- documented in the class docstring rather than left
  unexplained.

### FIX 4 (LOW, docs)

Removed the two stale/fragile line-number citations attempt 1 introduced (`play.py:73 ... at :518` and
`~line 200` in the new `bill_confirmed` docstring) -- both replaced with citations by NAME (`_run()`'s own
`self._tracker.tag(...)` call, `TeamTracker.tag`'s assignment) rather than a line number that already drifted
once between attempts. Also corrected the substance the verifier flagged: `_opp_elixir_v2_bill`'s docstring
now states explicitly that on the perception-thread ("fresh") path, the act loop's snapshot of `_team_tracker`
may be UP TO the ~2.0 s staleness ceiling play.py already tolerates for every other perception-fed consumer
(`fresh = age <= 2.0` in `_threat_extra`) -- not a new gap this function opens, but real latency worth
stating rather than implying the tag is always current. Kept the (correct) cross-references to
`pipeline/opp_est_audit.py`'s `tt_bill_dets` (line range) and the `filter_whitelisted_billed_dets`
docstring, unchanged, since the ticket said to.

### Verification run (attempt 2)

`cd icebow && PYTHONPATH=src ../icebow/.venv/Scripts/python.exe -m pytest tests/test_play_tracker_wiring.py
-v` -> **29 passed** (19 from attempt 1 plus 10 new: 5 `PerceptionBillConfirmedLockTests`, 5
`OppElixirV2ShadowWiring`, replacing the 1 weakened test FIX 3 flagged and rewriting
`test_record_my_play_is_unconditional_in_both_modes`).

Wider run: `tests/test_opponent_elixir.py tests/test_opponent_elixir_v2.py tests/test_play_no_cnn.py
tests/test_play_tracker_wiring.py tests/test_live_opponent.py tests/test_perception_with_base_i9.py
tests/test_reaction_and_phantoms.py tests/test_spell_phantom_credit.py tests/test_env_init_attrs.py -q` ->
**163 passed** (covers every test file that imports `clashrl.perception` or `clashrl.opponent_elixir`, plus
this ticket's own two files).

`ast.parse` on both edited files -> syntax OK. `git diff --stat` (icebow/src/clashrl/play.py,
icebow/src/clashrl/perception.py, icebow/tests/test_play_tracker_wiring.py only): 71/92/336 insertions
respectively, 4/0/0 deletions -- no other file touched (`git status --porcelain` on config.yaml,
opponent_elixir.py, replay_mine.py, pipeline/**, hogeq/** all clean).

### Corrected live-path matrix (opp_elixir_v2, opp_elixir_v2_shadow, student_opp_elixir)

| opp_elixir_v2 | shadow | student_opp_elixir | estimator(s) run | -> mem[5] | -> S1 (`_oe`) |
|---|---|---|---|---|---|
| F | F | F | V1 only, raw dets | V1 | None |
| F | F | T | V1 only, raw dets | V1 | V1's `_est` |
| F | T | F | V1 (raw dets) + V2 (tracker-billed, shadow) | V1 | None |
| F | T | T | V1 (raw dets) + V2 (tracker-billed, shadow) | V1 | V1's `_est` |
| T | F | F | V2 only, tracker-billed | V2 | None |
| T | F | T | V2 only, tracker-billed | V2 | V2's `_est` |
| T | T | F | V2 (tracker-billed) + V1 (raw dets, shadow) | V2 | None |
| T | T | T | V2 (tracker-billed) + V1 (raw dets, shadow) | V2 | V2's `_est` |

In every row: `_team_tracker`'s own construction/tagging/aim-assists/threat-identity vectors are untouched;
`record_my_play`/the match-reset call pattern are unconditional (both estimators reset when both exist);
the `detector_cards` whitelist applies to whatever feeds EITHER estimator (raw dets already filtered at the
existing site; tracker-billed dets filtered again on the tracker's output, same relative placement as
`pipeline/opp_est_audit.filter_whitelisted_billed_dets`). Shadow rows add exactly one extra `[play]` log
line per tick and the measured (isolated) ~2 microsecond/frame cost above; they never change `mem[5]` or
`_oe`.

### Exact revert (attempt 2)

`git checkout -- icebow/src/clashrl/play.py icebow/src/clashrl/perception.py
icebow/tests/test_play_tracker_wiring.py` (repo HEAD `ab76edb`, confirmed byte-identical to 3d8ad52 for
both source files) -- or restore byte-for-byte from `scratchpad/gauntlet/L67/opp_fix/backup/play.py.o19.orig`,
`perception.py.o19.orig`, and `test_play_tracker_wiring.py.o19.orig` (all three verified content-identical to
HEAD, CRLF-normalisation aside, before editing began each file). No config default was touched, so simply
never setting `play.opp_elixir_v2` / `play.opp_elixir_v2_shadow` in config.yaml is already a full
behavioural revert without touching code at all.

## STATUS: complete
