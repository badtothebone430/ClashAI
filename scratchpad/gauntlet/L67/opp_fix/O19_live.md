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

## STATUS: complete
