"""O8: offline harness measuring play.py's OpponentElixirEstimator against ENGINE ground truth, under THREE
input conditions, driven through the SAME engine match loop e1_eval uses (held-out ghost pool v1, live S1
policy). Does not modify pipeline/e1_eval.py, pipeline/e1_view.py, pipeline/obs_contract.py,
pipeline/engine_play.py, pipeline/e1_pool.py, or anything under icebow/src or hogeq/src -- every one of
those is IMPORTED unmodified. The per-tick loop below is a NEW instrumented copy of e1_eval.run_match's
while-loop (not a subclass -- run_match returns only match-level aggregates, no per-tick truth/estimate),
built entirely out of e1_eval's own pure helpers (obs_seed, _slot_maps, allowed_slots, anti_stall,
model_forward, live_decide) plus e1_pool.PoolV1Env, e1_view.live_view, obs_contract.from_engine/to_tokens,
and engine_play.compact_raw/cell_center/cell_to_engine/load_model -- all imported, none copied.

ATTEMPT 2 changes (progress file "## Attempt 2" has the full writeup):

CHANGE 1 -- estimator cadence. Live feeds ``_opp_elx.update()`` every detector/perception frame (~4-5 Hz),
NOT once per policy decision (0.5 s / 10 engine ticks) -- attempt 1 fed it at the (too coarse) decision
cadence, which the estimator's ``match_radius=0.07`` (~1.3 tiles) was never tuned to tolerate: a unit that
moves more than that between SAMPLES (not between real detector frames) loses its track and is double-charged
as a new play. Fix: step the engine in ``STEP_TICKS=5``-tick increments; call ``update()`` on every estimator
at EVERY 5-tick step; run the live policy only every ``decide_every`` (still 10) ticks, i.e. every OTHER step,
at the identical ticks attempt 1 (and e1_eval itself) decided on. The engine is a deterministic function of
history, so ``EngineMatchEnv._advance_to(target)`` (an absolute tick, stepping tick-by-tick and stopping
exactly on ghost ticks regardless of chunk size -- scratchpad/gauntlet/L62/engine_env.py:401-414) produces a
BIT-IDENTICAL state at tick 100 whether reached via one ``_advance_to(100)`` call or two 5-tick calls; the
smoke re-run confirms ``plays_accepted`` per match is unchanged from attempt 1 (see progress file).
RNG split this forces: e1_eval's own ``rng_obs`` (seeded ``obs_seed(tag, k)``) is consumed ONLY at policy
ticks, exactly as attempt 1 / e1_eval -- so the POLICY's decisions (and thus ``plays_accepted``) stay
byte-identical. Condition B needs a degraded view at the NON-policy 5-tick steps too, for which e1_eval has
no cadence of its own to borrow -- those extra draws come from a second, harness-local generator seeded
``b5_seed(tag, k)`` (module-level, same crc32 style as ``e1_eval.obs_seed``/``random_seed``, deliberately a
different domain string so it can never collide with e1_eval's own stream). This is the one place this
harness draws its own randomness; every other draw is e1_eval's.

CHANGE 2 -- condition A+ (``est_Aplus``): perfect detection INCLUDING spells (``bs.spells``, not just
``bs.units``). Quantifies part of the blind spot attempt 1 found: play.py:547 pre-filters ``dets`` to
``team == "enemy" and base in observation.detector_cards`` before ``update()`` ever sees them. A (units only,
NO whitelist applied -- see ATTEMPT 4's correction below, this claim was wrong in earlier attempts) is kept
as an upper-bound ceiling; A+ (units + spells, also no whitelist) is the strictly-better ceiling if spells
were ever priced in at all. A spell whose CardDB cost is None (measured in attempt 1: Mirror -- dynamic
pricing, no flat ``elixir`` key) is skipped from A+'s dets (not charged as a free play) and logged ONCE per
base name per process, per the ticket.

Per-tick rows now carry ``truth_drop`` (attempt 1 folded this into a boolean; the raw size makes multi-play
aliasing inside one decision window visible instead of hidden -- see progress file). Summaries add
``bias`` (not just MAE) at opponent-play ticks, and an MAE split by ``truth >= 9.0`` (near the [0,10] cap,
where the counting identity is expected to strain) vs below.

ATTEMPT 3 (progress file "## Attempt 3" has the full writeup): diagnostic for the coordinator's
spawned-units hypothesis (attempt 2's A_bias moved -3.42 -> -3.34 under the cadence fix, i.e. barely, so
track loss from a too-coarse sample rate is not the dominant cause of the persistent NEGATIVE bias; a
negative bias under ``est = my_elixir + my_spent - opp_spent`` with PERFECT detection means ``opp_spent`` is
counting more than the opponent actually spent -- something is being charged that should not be). Two new
per-condition ledgers:
  ``charged_by_base`` -- every base the estimator's own ``_opp_spent`` accounting actually charged, and how
    much. Built by ``TrackedEstimator``, a subclass that does NOT reimplement or duplicate any of
    ``OpponentElixirEstimator``'s tracking/clustering logic (which would risk drifting from the real
    algorithm): ``update()`` is the ONLY place ``_opp_spent`` is ever incremented from a detection (the
    saturation rebase at lines 122-129 is a bookkeeping correction, not attributable to a specific card), and
    the ONLY thing that method reads to price a new track is exactly one call, ``self.db.elixir(base)``
    (opponent_elixir.py:109). ``TrackedEstimator.update()`` temporarily swaps ``self.db`` for a one-shot
    recorder proxy for the duration of the (unmodified) ``super().update()`` call, restores the real ``db``
    immediately after, and attributes every call the REAL charging code made (base, cost) with cost > 0 to
    ``charged_by_base``. This is the faithful option over diffing ``_opp_spent`` before/after: a diff cannot
    tell two different bases apart when both become new tracks inside the SAME ``update()`` call (a common
    case once ``update()`` batches several 5-tick-cadence detections), while wrapping the collaborator
    captures the real per-base charge event with no re-implementation and no ambiguity.
  ``ghost_played_by_base`` -- the TRUTH ledger: the opponent's actual card plays, from ``entry["ghost_commands"]``
    (the exact list ``PoolV1Mixin``/``EngineMatchEnv`` re-drives in eval mode -- e1_pool.py's own ``_cmd()``
    rows), excluding ability entries (``card`` is ``None``), costed via the SAME ``CardDB`` (never the
    engine's own recorded ``cost`` field, so both ledgers price off one source) after normalizing the pool's
    hyphenated slugs (``"giant-snowball"``) to vocab's underscore convention and folding evo/hero suffixes
    with ``vocab.base_key`` -- the SAME base-key space ``charged_by_base`` uses, so the two ledgers are
    directly comparable base-for-base.
Summary adds, per condition: an over-charge table (base | ghost_plays | ghost_elixir | charges |
charged_elixir | over_charge, top 25 by |over_charge| absolute value), ``total_over_charge``, and
``over_charge_share_from_bases_never_played`` (the fraction of charged elixir landing on bases the ghost
NEVER played at all -- a spawned body, e.g. tombstone's skeletons or a witch's summons, is BY DEFINITION not
in ``ghost_deck``/``ghost_commands`` since only the 8 played cards are, so this number directly tests the
spawned-units hypothesis). Per-match: ``over_charge_at_end`` (total charged elixir minus total ghost elixir
for that match) per condition, in each match's summary and in ``summary.json``'s
``over_charge_at_end_by_match``.

ATTEMPT 4 (progress file "## Attempt 4" has line numbers for every fix; blind verifier report
scratchpad/gauntlet/L67/opp_est/V8_verify.md, baseline 06dcc6d):

FIX 1 (the verifier's FAIL) -- the over-charge truth ledger was built from ``entry["ghost_commands"]``, the
SCRIPT, but the verifier measured (ctrl_live100, 100 matches) only 2429 of 3796 scripted ghost plays were
ever DELIVERED (68 refused, 1299 undelivered because 62/100 matches end before the script does) -- biasing
every over-charge figure NEGATIVE by about a third of ghost elixir, directly confounding the diagnosis. Fixed
by reading ``env.ghost_cards_delivered`` (a ``collections.Counter`` keyed by slug; ``PoolV1Mixin.reset()``
zeroes it per match at e1_pool.py:519, ``_fire_ghosts_at`` increments it ONLY on an accepted, non-mine ghost
play at e1_pool.py:562) at the end of each match and costing THAT via CardDB -- ``ghost_delivered_by_base``.
The old function is kept as ``ghost_scripted_by_base`` (same source, same costing) for reference ONLY,
clearly labelled in the summary and NEVER fed into ``over_charge``/``total_over_charge``/
``over_charge_share_from_bases_never_played``/``over_charge_at_end``, all of which now read exclusively from
``ghost_delivered_by_base``.

FIX 2 -- ``is_opp_play_tick`` (truth-drop >= 1.0) silently misses every 1-cost ghost play (regen offsets a
1-elixir drop below the threshold; the verifier measured 1613/12205 = 13% of held-out ghost plays cost 1).
Replaced with ``opp_play_flags``, a pure function over (a) the ascending list of this match's sampled ticks
and (b) ``env.ghost_events`` (a ``(tick, accepted 0/1, reason)`` log the SAME ghost-drive path already keeps,
engine_env.py:334 initializes it empty per match, PoolV1Mixin._fire_ghosts_at appends to it on every ghost
play attempt whether accepted or refused): the FIRST sampled tick at or after each ACCEPTED delivery's
recorded tick is flagged. CAVEAT, stated plainly: the recorded tick is the play's ORIGINAL SCRIPTED tick
(``g["tick"]``), not necessarily the engine tick it was actually accepted at -- a `not_enough_elixir` refusal
retries up to ``elixir_slack`` (40 ticks / 2s) later before either succeeding or being counted as refused
(e1_pool.py's ``_fire_ghosts_at``). Since the real accept tick is always >= the recorded one, flagging the
first sample at-or-after the recorded tick never MISSES a delivery, but for a retried play it can flag one or
two samples slightly EARLY (before the unit is actually visible on board). ``truth_drop`` (attempt 3) is kept
as an informational column, no longer used to set the flag.

FIX 3 -- per-match ``outcome``/``crowns_for``/``crowns_against`` (``ep._outcome(env, state)``'s second return
value was being discarded), ``plays_accepted``, ``decisions``, ``end_tick``, ``n_ticks`` are now persisted in
``summary.json``'s ``matches: [...]`` and in the final JSON line, field-for-field the same spelling
``e1_eval.run_match`` writes to ``matches.jsonl`` (``outcome`` a "win"/"draw"/"loss" string, ``crowns_for``/
``crowns_against`` ints, ``plays_accepted`` the accepted-play count) -- so the lead can diff this harness's
matches directly against ``attrib_noise/ctrl_live100/slot0/matches.jsonl`` as a fidelity gate.

FIX 4 -- condition ``A_wl``: perfect detection, enemy units only, WITH live's ``observation.detector_cards``
whitelist applied (``load_detector_cards``, loads ``icebow/config/config.yaml`` through the SAME unmodified
``clashrl.config.Config`` play.py itself uses -- ``cfg.get("observation", "detector_cards", default=[])``,
play.py:376). This is live's ACTUAL structural ceiling (A, kept unchanged, is a looser upper bound with no
whitelist at all -- the earlier "100% troop/building, zero spell entries" claim in CHANGE 2 above was WRONG
and is corrected here). MEASURED, independently re-derived from the pool file (not just taken from the
verifier): the whitelist has 46 entries; ``the_log`` (a spell -- ``vocab.SPELL_CLASSES`` -- config.yaml:494)
IS one of them, so "zero spell entries" was false. Across the held-out split's 12205 non-ability ghost card
plays, 4202 (34.4%) are non-spell bases the whitelist excludes (e.g. archer_queen, golem, mega_minion); of
the 9614 non-spell plays specifically, that is 4202/9614 = 43.7% -- both framings are given here since the
verifier's "34%" used the ALL-plays denominator, not the non-spell one. Summaries now report A+, A, A_wl, B.

NOTE (no fix needed, flagged by the verifier): ``--max-ticks`` truncates each match at that row count, and
``ep._outcome`` is then computed on a NON-TERMINAL engine state -- the CLI help now says this is smoke-only.

    icebow/.venv/Scripts/python.exe -m pipeline.opp_est_audit --port 38032 --ckpt <pt> --split heldout ^
        --entries 0:2 --max-ticks 600 --out scratchpad/gauntlet/L67/opp_est/smoke

TICKET O10 -- charge-event TRACE (no fix; instrumentation only). ATTEMPT 4's over-charge tables (real audit,
condition A) found continuously-visible SINGLE units re-billed many times over (tombstone: 58 delivered plays
-> 442 charges; miner 28 -> 201; witch 16 -> 142) despite ``_find_track`` (opponent_elixir.py:101-105)
appearing, by reading, to refresh a matched track's timestamp every update rather than re-charging it. This
adds ``--charge-trace``: for condition A ONLY, ``<out>/charges.jsonl`` (one row per actual ``_opp_spent``
increment, with the diagnostics needed to tell WHY the charge happened) and ``<out>/track_events.jsonl`` (one
row per track EXPIRY, i.e. every base ``_seen_tracks`` -- opponent_elixir.py:42-43 -- actually removes).

``TracedEstimator(TrackedEstimator)`` gets both WITHOUT re-implementing any tracking/matching/clustering
logic and WITHOUT touching opponent_elixir.py: it overrides ``_seen_tracks`` to diff ``self._tracks``
before/after the real (unmodified) ``super()._seen_tracks(now)`` call -- the ticket's own "diff before/after"
option, chosen over "replicate the same filter" because a diff can never drift from the real ``forget_s``
comparison if that constant or the comparison operator ever changes. ``update()`` snapshots ``self._tracks``
by object identity BEFORE calling ``super().update()`` (which runs the real, unmodified expiry -> match ->
cluster -> charge pipeline), then reads off which tracks in the POST-update list are new objects (never
mutated in place, only ever created fresh inside ``_cluster_new``'s loop, opponent_elixir.py:107-108) and
zips them 1:1, in order, against ``TrackedEstimator``'s own ``_RecorderDB`` call list (order-preserved because
both the ``self._tracks.append(...)`` and the ``self.db.elixir(base)`` call sit in the SAME loop iteration,
opponent_elixir.py:107-111) to recover (base, x, y, cost) for every actual charge, cost > 0 only, mirroring
``charged_by_base``'s own gate exactly.

Per-charge diagnostics are computed by the harness (``charge_diagnostics``), never inside the estimator, from
two things the estimator itself does not remember past one call: the PREVIOUS sample's det list (caller-
tracked, one match at a time) and the pre-update track snapshot ``TracedEstimator.update`` already captured
for its own bookkeeping. ``classify_charge_reason`` turns those into a best-effort ``reason`` string; see the
module's inline comment on that function for the one place its evaluation order deliberately departs from the
ticket text's literal listing order (and why -- the "rebill_after_expiry" acceptance scenario forces it).
Full writeup: scratchpad/gauntlet/L67/opp_fix/O10_trace.md.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_ICEBOW_SRC = REPO / "icebow" / "src"
if str(_ICEBOW_SRC) not in sys.path:
    sys.path.insert(0, str(_ICEBOW_SRC))

from pipeline import vocab                                                         # noqa: E402
from pipeline.dataset import _past                                                 # noqa: E402
from pipeline.e1_eval import (                                                     # noqa: E402
    DECIDE_EVERY, MAX_U, SLOT_OF_PORT, STALL_ELIXIR_LIVE, STALL_SECONDS_LIVE, TAU_LIVE,
    _slot_maps, allowed_slots, anti_stall, live_decide, make_tasks, model_forward, obs_seed,
    parse_entries, parse_seeds, parse_shard, refuse_existing_out,
)
from pipeline.e1_pool import POOL_V1, PoolV1Env, load_pool_v1, select_split, sha256_file  # noqa: E402
from pipeline.e1_view import Noise, live_view                                      # noqa: E402
from pipeline.obs_contract import _phase, from_engine, load_deck, to_tokens        # noqa: E402
from pipeline import engine_play as ep                                             # noqa: E402

from clashrl.cards import CardDB                                                   # noqa: E402
from clashrl.config import Config                                                  # noqa: E402
from clashrl.opponent_elixir import OpponentElixirEstimator                        # noqa: E402


STEP_TICKS = 5              # CHANGE 1: estimator update cadence (engine ticks); must divide --decide-every
NEAR_CAP_TRUTH = 9.0         # CHANGE 2 (summary split): truth >= this is "near the [0, 10] saturation cap"


def b5_seed(tag: str, k: int) -> int:
    """Harness-local RNG domain for Condition B's estimator updates at the NON-policy 5-tick steps (module
    docstring CHANGE 1). Same crc32(f"{tag}:<domain>:{k}") shape as e1_eval.obs_seed/random_seed, a different
    domain string so it can never collide with either of those."""
    return zlib.crc32(f"{tag}:oppest_b5:{k}".encode())


def is_policy_tick(tick: int, first_tick: int, decide_every: int) -> bool:
    """True on the ticks the live policy actually decides on: the first decision tick and every
    ``decide_every`` engine ticks after it -- e1_eval's own cadence, unchanged by the finer 5-tick stepping."""
    return (int(tick) - int(first_tick)) % int(decide_every) == 0


# ------------------------------------------------------------------------------------------------------
# detection shim -- the shape OpponentElixirEstimator.update() reads (.base, .cx, .gy, .team); mirrors
# scratchpad/gauntlet/L67/opp_estimator_sim.py's ``_Det`` (that harness is the SIM, not the engine -- copied
# for its shape only, per the ticket).
# ------------------------------------------------------------------------------------------------------
class _Det:
    __slots__ = ("base", "cx", "gy", "team")

    def __init__(self, base: str, cx: float, gy: float, team: str):
        self.base = str(base)
        self.cx = float(cx)
        self.gy = float(gy)
        self.team = str(team)


_TEAM_OF_SIDE = {0: "mine", 1: "enemy", -1: "unknown"}
PHASES = ("single", "double", "overtime")


def dets_of(items) -> list[_Det]:
    """Any ``BoardState`` unit sequence (``.units`` OR ``.spells``) -> ``_Det`` list, every side kept (the
    estimator's own ``update()`` filters to ``team == "enemy"`` at opponent_elixir.py:94, so passing all
    three and letting it filter is equivalent to pre-filtering, and matches e1_eval's own dets shape)."""
    return [_Det(vocab.base_key(vocab.UNIT_VOCAB[u.cls]), u.x, u.y, _TEAM_OF_SIDE.get(u.side, "unknown"))
            for u in items]


def dets_of_costed(items, db: CardDB, warned: set) -> list[_Det]:
    """Like ``dets_of`` but drops (never zero-costs) any item whose base has no CardDB elixir row, logging
    the base ONCE per process into ``warned`` -- CHANGE 2's A+ spell path (e.g. Mirror: dynamic pricing, no
    flat ``elixir`` key, found in attempt 1's dets-conversion test)."""
    out = []
    for u in items:
        base = vocab.base_key(vocab.UNIT_VOCAB[u.cls])
        if db.elixir(base) is None:
            if base not in warned:
                warned.add(base)
                print(f"[opp_est_audit] A+: {base!r} has no CardDB elixir cost -- skipped from dets "
                     f"(not charged as free)", flush=True)
            continue
        out.append(_Det(base, u.x, u.y, _TEAM_OF_SIDE.get(u.side, "unknown")))
    return out


def load_detector_cards(config_path) -> frozenset[str]:
    """FIX 4: live's ``observation.detector_cards`` whitelist, loaded the SAME way play.py:376 does
    (``cfg.get("observation", "detector_cards", default=[])``) through the UNMODIFIED ``clashrl.config.
    Config`` -- never a re-parse of the yaml by hand. ``config_path`` is ``deck.config``
    (``icebow/config/config.yaml``, per pipeline/decks/icebow.yaml)."""
    cfg = Config.load(str(config_path))
    return frozenset(str(c) for c in (cfg.get("observation", "detector_cards", default=[]) or []))


def dets_of_whitelisted(items, whitelist: frozenset) -> list[_Det]:
    """Like ``dets_of`` but drops an ENEMY det whose base is outside live's ``detector_cards`` whitelist
    (FIX 4's Condition A_wl, live's actual structural ceiling) -- mine/unknown dets pass through unfiltered
    since the estimator's own ``update()`` drops them by team regardless (opponent_elixir.py:94)."""
    return [d for d in dets_of(items) if d.team != "enemy" or d.base in whitelist]


def n_enemy(items) -> int:
    return sum(1 for u in items if u.side == 1)


def phase_from_flags(double_elixir: bool, overtime: bool) -> str:
    if overtime:
        return "overtime"
    if double_elixir:
        return "double"
    return "single"


def phase_of(bs) -> str:
    return phase_from_flags(bs.double_elixir, bs.overtime)


def card_cost(db: CardDB, cls_id: int) -> Optional[int]:
    """Vocab id -> elixir cost via the same base-key fold the estimator's own ``db.elixir(d.base)`` call uses."""
    return db.elixir(vocab.base_key(vocab.UNIT_VOCAB[cls_id]))


# ------------------------------------------------------------------------------------------------------
# ATTEMPT 3: charge attribution -- see the module docstring for why db-wrapping (not a diff of _opp_spent,
# not a reimplementation of update()) is the faithful way to get this without touching opponent_elixir.py.
# ------------------------------------------------------------------------------------------------------
class _RecorderDB:
    """One-shot proxy: forwards ``.elixir(base)`` to the REAL db and records every (base, cost) call made
    while it is installed. Nothing else of CardDB is used inside ``OpponentElixirEstimator.update()``, so
    forwarding only ``elixir`` is sufficient (an AttributeError on anything else would mean update() started
    reading something new from db and this proxy needs a second look, which is exactly the failure mode we
    WANT if opponent_elixir.py ever changes underneath this harness)."""

    def __init__(self, real_db, calls: list):
        self._real = real_db
        self._calls = calls

    def elixir(self, base):
        c = self._real.elixir(base)
        self._calls.append((base, c))
        return c


class TrackedEstimator(OpponentElixirEstimator):
    """``OpponentElixirEstimator`` with a per-instance ``charged_by_base`` ledger, built by intercepting the
    ONE ``self.db.elixir(base)`` call ``update()`` makes per new track (opponent_elixir.py:109) -- never by
    re-deriving which dets are "new" ourselves (that would duplicate ``_find_track``/``_cluster_new`` and
    could drift from the real matching/clustering radii)."""

    def __init__(self, db):
        super().__init__(db)
        self.charged_by_base: dict[str, list] = {}   # base -> [n_charges, total_elixir_charged]
        self.last_calls: list[tuple[str, Optional[float]]] = []   # O10: this update() call's raw db.elixir()
                                                                    # calls (base, cost), in _cluster_new order --
                                                                    # read-only bookkeeping, no behavior change.

    def update(self, my_elixir, enemy_dets, now):
        real_db = self.db
        calls: list[tuple[str, Optional[float]]] = []
        self.db = _RecorderDB(real_db, calls)
        try:
            result = super().update(my_elixir, enemy_dets, now)
        finally:
            self.db = real_db
        for base, c in calls:
            cf = float(c or 0.0)
            if cf > 0.0:                              # mirrors update()'s own ``if c > 0.0`` charge gate exactly
                row = self.charged_by_base.setdefault(base, [0, 0.0])
                row[0] += 1
                row[1] += cf
        self.last_calls = calls        # O10: exposed so TracedEstimator can pair (base, cost) with new tracks
        return result


# ------------------------------------------------------------------------------------------------------
# TICKET O10: charge-event / track-expiry trace -- see module docstring for the full design rationale.
# ------------------------------------------------------------------------------------------------------
class TracedEstimator(TrackedEstimator):
    """``TrackedEstimator`` plus two mechanical, read-only observations of the SAME unmodified super() calls:
    ``last_charges`` (this update() call's newly-tracked, cost > 0 bases with (x, y, cost, n_tracks_before,
    n_tracks_after)) and ``last_expired`` (this update() call's track EXPIRIES, i.e. every track
    ``_seen_tracks`` actually dropped). Neither re-implements matching/clustering/forgetting -- both are
    diffs of ``self._tracks`` around the real, unmodified super() calls, by object identity (track dicts are
    mutated in place when refreshed, and are only ever created fresh by ``_cluster_new``'s loop, so identity
    alone tells 'refreshed' from 'new' or 'gone' correctly)."""

    def __init__(self, db):
        super().__init__(db)
        self.last_charges: list[dict] = []
        self.last_expired: list[dict] = []

    def _seen_tracks(self, now: float) -> None:
        before = list(self._tracks)
        super()._seen_tracks(now)                      # the REAL, unmodified forget_s filter
        kept_ids = {id(tr) for tr in self._tracks}
        self.last_expired = [
            {"base": tr["base"], "x": tr["x"], "y": tr["y"], "age_s": float(now) - float(tr["t"]),
             "last_seen_t": float(tr["t"])}
            for tr in before if id(tr) not in kept_ids
        ]

    def update(self, my_elixir, enemy_dets, now):
        tracks_before = list(self._tracks)             # BEFORE this call's _seen_tracks/match/cluster/charge
        before_ids = {id(tr) for tr in tracks_before}
        n_before = len(tracks_before)
        result = super().update(my_elixir, enemy_dets, now)   # real _seen_tracks (via our override above) +
                                                                 # real match/cluster/charge (via TrackedEstimator)
        new_tracks = [tr for tr in self._tracks if id(tr) not in before_ids]   # _cluster_new append order
        n_after = len(self._tracks)
        charges = []
        for tr, (base, cost) in zip(new_tracks, self.last_calls):
            cf = float(cost or 0.0)
            if cf > 0.0:                                # same charge gate as charged_by_base
                charges.append({"base": tr["base"], "x": tr["x"], "y": tr["y"], "cost": cf,
                                "n_tracks_before": n_before, "n_tracks_after": n_after})
        self.last_charges = charges
        return result


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def charge_diagnostics(base: str, x: float, y: float, prev_dets: Optional[list], tracks_before: list[dict],
                       match_radius: float, cluster_radius: float) -> dict:
    """O10 (b): the diagnostic fields computed by the HARNESS (never inside the estimator) from the PREVIOUS
    sample's dets and the pre-update track snapshot ``TracedEstimator.update`` already took."""
    prev_same = [d for d in (prev_dets or [])
                if str(getattr(d, "team", None)) == "enemy" and str(getattr(d, "base", "")) == base]
    prev_dists = [_dist(float(d.cx), float(d.gy), x, y) for d in prev_same]
    prev_min = min(prev_dists) if prev_dists else None
    track_same = [tr for tr in tracks_before if tr["base"] == base]
    track_dists = [_dist(float(tr["x"]), float(tr["y"]), x, y) for tr in track_same]
    track_min = min(track_dists) if track_dists else None
    return {
        "prev_same_base_within_match_radius": bool(prev_min is not None and prev_min <= match_radius),
        "prev_same_base_min_dist": (round(prev_min, 4) if prev_min is not None else None),
        "prev_n_same_base": len(prev_same),
        "same_base_track_existed_before": bool(track_same),
        "nearest_track_same_base_dist": (round(track_min, 4) if track_min is not None else None),
    }


def classify_charge_reason(base: str, diag: dict, expired_bases_this_update: set, match_radius: float,
                           cluster_radius: float) -> str:
    """O10 (c): best-effort classification from ``charge_diagnostics`` + this SAME update() call's
    track_events (``expired_bases_this_update``, from ``TracedEstimator.last_expired``).

    DEVIATION FROM THE TICKET TEXT'S LITERAL LISTING ORDER (first_seen, split, rebill_after_expiry,
    rebill_out_of_radius, other), flagged in the progress file and hand-back: this checks
    rebill_after_expiry BEFORE split. The ticket's own acceptance scenario -- a unit absent 7s then
    reappearing at (or near) its old, stationary position -- is simultaneously <= cluster_radius of its own
    previous-sample detection (it never moved) AND had a live track before this update, so under the literal
    a/b/c/d order it would hit 'split' first and never reach 'rebill_after_expiry', contradicting that same
    acceptance test. Checking the expiry cross-reference first resolves this in the direction the acceptance
    test requires, and is the more causally direct story (the track was PROVABLY just removed this exact
    update) versus a merely coincidental positional match. All raw diagnostic fields are kept on every row
    regardless, so the lead can re-derive any other ordering they prefer."""
    if diag["prev_n_same_base"] == 0 and not diag["same_base_track_existed_before"]:
        return "first_seen"
    if diag["same_base_track_existed_before"] and base in expired_bases_this_update:
        return "rebill_after_expiry"
    if (diag["prev_same_base_min_dist"] is not None and diag["prev_same_base_min_dist"] <= cluster_radius
            and diag["same_base_track_existed_before"]):
        return "split"
    if diag["prev_n_same_base"] > 0 and (diag["prev_same_base_min_dist"] is None
                                         or diag["prev_same_base_min_dist"] > match_radius):
        return "rebill_out_of_radius"
    return "other"


def run_traced_update(est: TracedEstimator, my_elixir: float, dets: list, now: float, tick: int, tag: str,
                      k: int, prev_dets: Optional[list], prev_t: Optional[float], t_to_tick: dict
                      ) -> tuple[list[dict], list[dict]]:
    """One condition-A ``TracedEstimator.update()`` cycle with tracing: calls the real (unmodified) update(),
    then builds this call's charge rows (module docstring O10 (a)/(b)/(c)) and track-event rows from
    ``est.last_charges``/``est.last_expired`` plus ``prev_dets`` (the PREVIOUS sample's det list for this
    match, caller-tracked) and ``t_to_tick`` (this match's t_sec -> tick map, caller-updated every sample so
    a track's stored ``t`` -- always some earlier call's ``now`` -- can be resolved back to a tick)."""
    tracks_before = [dict(tr) for tr in est._tracks]        # value snapshot, immune to later in-place mutation
    t_to_tick[float(now)] = int(tick)
    est.update(my_elixir, dets, now)
    expired_bases = {ex["base"] for ex in est.last_expired}
    match_radius, cluster_radius = est.match_radius, est.cluster_radius
    charge_rows = []
    for ch in est.last_charges:
        diag = charge_diagnostics(ch["base"], ch["x"], ch["y"], prev_dets, tracks_before,
                                  match_radius, cluster_radius)
        reason = classify_charge_reason(ch["base"], diag, expired_bases, match_radius, cluster_radius)
        dt = (round(float(now) - float(prev_t), 4) if prev_t is not None else None)
        charge_rows.append({"tag": tag, "k": int(k), "tick": int(tick), "t_sec": round(float(now), 3),
                            "base": ch["base"], "cost": ch["cost"], "x": ch["x"], "y": ch["y"],
                            "n_tracks_before": ch["n_tracks_before"], "n_tracks_after": ch["n_tracks_after"],
                            "reason": reason, "dt_since_prev_sample_s": dt, **diag})
    track_rows = []
    for ex in est.last_expired:
        track_rows.append({"tag": tag, "k": int(k), "tick": int(tick), "base": ex["base"], "x": ex["x"],
                           "y": ex["y"], "age_s": round(float(ex["age_s"]), 4),
                           "last_seen_tick": t_to_tick.get(ex["last_seen_t"])})
    return charge_rows, track_rows


def _base_of_slug(slug: str) -> str:
    """The pool's hyphenated slugs (e.g. ``"giant-snowball"``) normalized to vocab's underscore convention,
    then evo/hero suffix-folded -- the SAME base-key space ``charged_by_base`` uses."""
    return vocab.base_key(str(slug).replace("-", "_"))


def _cost_ledger_add(out: dict[str, list], base: str, n: int, cost: float) -> None:
    row = out.setdefault(base, [0, 0.0])
    row[0] += n
    row[1] += n * cost


def _warn_costless_once(base: str, warned: set, where: str) -> None:
    if base not in warned:
        warned.add(base)
        print(f"[opp_est_audit] {where}: {base!r} has no CardDB elixir cost -- excluded from the ledger "
             f"(not charged as free)", flush=True)


def ghost_scripted_by_base(entry: dict, db: CardDB, warned: Optional[set] = None) -> dict[str, list]:
    """REFERENCE ONLY (FIX 1 -- see module docstring): the opponent's SCRIPTED card plays for one
    pool entry, from ``entry["ghost_commands"]`` (ability entries excluded -- ``card`` is ``None`` for those,
    e1_pool.py's ``_cmd()``), costed via CardDB. NOT what actually happened in the match -- a scripted play
    can be refused, or the match can end before the script does (verifier: ~1/3 of scripted plays, ctrl_
    live100). Use ``ghost_delivered_by_base`` for anything measuring the estimator against truth; this
    function exists only so the SCRIPT is visible for reference, clearly labelled as such in the summary."""
    if warned is None:
        warned = set()
    out: dict[str, list] = {}
    for c in entry.get("ghost_commands") or []:
        if c.get("ability") or not c.get("card"):
            continue
        base = _base_of_slug(c["card"])
        cost = db.elixir(base)
        if cost is None:
            _warn_costless_once(base, warned, "ghost_scripted_by_base")
            continue
        _cost_ledger_add(out, base, 1, float(cost))
    return out


def ghost_delivered_by_base(delivered: dict, db: CardDB, warned: Optional[set] = None) -> dict[str, list]:
    """FIX 1 -- the TRUTH ledger every over-charge figure uses: the opponent's ACTUALLY DELIVERED card plays
    for one match, from ``env.ghost_cards_delivered`` (a ``collections.Counter`` keyed by slug; PoolV1Mixin.
    reset() zeroes it per match at e1_pool.py:519, ``_fire_ghosts_at`` increments it ONLY on an accepted,
    non-mine ghost play at e1_pool.py:562 -- refused and never-reached plays are never counted), costed via
    the SAME CardDB / base-key folding ``ghost_scripted_by_base``/``charged_by_base`` use. ``delivered`` is
    ``dict(env.ghost_cards_delivered)`` (or any ``{slug: count}`` mapping -- a hand-built stub for tests)."""
    if warned is None:
        warned = set()
    out: dict[str, list] = {}
    for slug, n in (delivered or {}).items():
        if not slug or not n:
            continue
        base = _base_of_slug(slug)
        cost = db.elixir(base)
        if cost is None:
            _warn_costless_once(base, warned, "ghost_delivered_by_base")
            continue
        _cost_ledger_add(out, base, int(n), float(cost))
    return out


def opp_play_flags(sample_ticks: list, ghost_events: list) -> list:
    """FIX 2 -- one bool per ``sample_ticks`` entry (ascending engine ticks, this match's per-row ticks):
    True on the FIRST sample tick at or after each ACCEPTED ghost delivery's recorded tick. ``ghost_events``
    is ``env.ghost_events`` for this match, ``(tick, accepted 0/1, reason)`` tuples -- engine_env.py:334
    initializes it to ``[]`` per match reset, PoolV1Mixin._fire_ghosts_at appends one entry per ghost play
    ATTEMPT (accepted or refused); only ``accepted == 1`` entries count as a delivery here. Multiple
    deliveries landing between the same two sample ticks all flag the one next sample tick they share.
    CAVEAT (module docstring FIX 2): the recorded tick is the play's ORIGINAL SCRIPTED tick, which for a
    retried play (elixir-slack) can be up to ``elixir_slack`` ticks before the real accept tick -- this can
    flag a sample slightly EARLY but never misses a delivery, since the real accept tick is always >= it."""
    delivered = sorted(int(t) for t, ok, _r in ghost_events if int(ok) == 1)
    out = [False] * len(sample_ticks)
    di = 0
    for i, t in enumerate(sample_ticks):
        while di < len(delivered) and delivered[di] <= int(t):
            out[i] = True
            di += 1
    return out


def merge_base_ledgers(ledgers: list[dict]) -> dict[str, list]:
    """Sum a list of {base: [n, total]} dicts (one per match) into one run-level ledger."""
    out: dict[str, list] = {}
    for d in ledgers:
        for base, (n, tot) in d.items():
            row = out.setdefault(base, [0, 0.0])
            row[0] += n
            row[1] += tot
    return out


def overcharge_table(charged: dict[str, list], ghost: dict[str, list], top: int = 25) -> dict:
    """Per-condition base | ghost_plays | ghost_elixir | charges | charged_elixir | over_charge table (module
    docstring, ATTEMPT 3), sorted by |over_charge| descending, plus the two run-level diagnostics."""
    bases = set(charged) | set(ghost)
    rows = []
    for b in sorted(bases):
        gp, ge = ghost.get(b, [0, 0.0])
        cp, ce = charged.get(b, [0, 0.0])
        rows.append({"base": b, "ghost_plays": int(gp), "ghost_elixir": round(float(ge), 2),
                    "charges": int(cp), "charged_elixir": round(float(ce), 2),
                    "over_charge": round(float(ce) - float(ge), 2)})
    rows.sort(key=lambda r: -abs(r["over_charge"]))
    total_charged = sum(r["charged_elixir"] for r in rows)
    total_ghost = sum(r["ghost_elixir"] for r in rows)
    never_played_charged = sum(r["charged_elixir"] for r in rows if r["ghost_plays"] == 0)
    share = (never_played_charged / total_charged) if total_charged > 0 else None
    return {"top": rows[:top], "n_bases": len(rows),
           "total_charged_elixir": round(total_charged, 2), "total_ghost_elixir": round(total_ghost, 2),
           "total_over_charge": round(total_charged - total_ghost, 2),
           "over_charge_share_from_bases_never_played": (round(share, 4) if share is not None else None)}


# ------------------------------------------------------------------------------------------------------
# pure metric helpers (tested offline; no numpy percentile surprises -- linear interpolation, matches
# numpy.percentile's default so results are reproducible without importing it in the hot path)
# ------------------------------------------------------------------------------------------------------
def mae(errs: list[float]) -> Optional[float]:
    return sum(abs(e) for e in errs) / len(errs) if errs else None


def mean_bias(errs: list[float]) -> Optional[float]:
    return sum(errs) / len(errs) if errs else None


def percentile(vals: list[float], q: float) -> Optional[float]:
    """Linear-interpolation percentile (numpy.percentile's default method), q in [0, 100]."""
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return float(s[0])
    rank = (q / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def share_within(abs_errs: list[float], bound: float) -> Optional[float]:
    return sum(1 for e in abs_errs if e <= bound) / len(abs_errs) if abs_errs else None


def summarize_condition(rows: list[dict], est_key: str) -> dict:
    """One condition's ("Aplus"/"A"/"B") full metric block over ``rows`` (dicts with truth/est*/phase/tag/k/
    is_opp_play_tick/is_my_play_tick)."""
    errs = [r[est_key] - r["truth"] for r in rows]
    abs_errs = [abs(e) for e in errs]
    out: dict[str, Any] = {
        "n_ticks": len(rows),
        "mae": mae(errs), "mean_bias": mean_bias(errs), "p90_abs_err": percentile(abs_errs, 90),
        "share_abs_err_le_1_0": share_within(abs_errs, 1.0), "share_abs_err_le_2_0": share_within(abs_errs, 2.0),
        "by_phase": {}, "opp_play_ticks": {}, "my_play_ticks": {}, "by_match": {}, "by_truth_cap": {},
    }
    for ph in PHASES:
        sub = [r[est_key] - r["truth"] for r in rows if r["phase"] == ph]
        out["by_phase"][ph] = {"n": len(sub), "mae": mae(sub), "mean_bias": mean_bias(sub),
                               "p90_abs_err": percentile([abs(e) for e in sub], 90)}
    opp_sub = [r[est_key] - r["truth"] for r in rows if r.get("is_opp_play_tick")]
    out["opp_play_ticks"] = {"n": len(opp_sub), "mae": mae(opp_sub), "bias": mean_bias(opp_sub)}
    my_sub = [r[est_key] - r["truth"] for r in rows if r.get("is_my_play_tick")]
    out["my_play_ticks"] = {"n": len(my_sub), "mae": mae(my_sub), "bias": mean_bias(my_sub)}
    near_sub = [r[est_key] - r["truth"] for r in rows if r["truth"] >= NEAR_CAP_TRUTH]
    below_sub = [r[est_key] - r["truth"] for r in rows if r["truth"] < NEAR_CAP_TRUTH]
    out["by_truth_cap"] = {
        "near_cap_ge_9": {"n": len(near_sub), "mae": mae(near_sub), "mean_bias": mean_bias(near_sub)},
        "below_cap": {"n": len(below_sub), "mae": mae(below_sub), "mean_bias": mean_bias(below_sub)},
    }
    by_match: dict[tuple, list[float]] = {}
    for r in rows:
        by_match.setdefault((r["tag"], r["k"]), []).append(abs(r[est_key] - r["truth"]))
    match_maes = [sum(v) / len(v) for v in by_match.values() if v]
    out["by_match"] = {"n_matches": len(by_match), "mae_median": (statistics.median(match_maes)
                        if match_maes else None), "mae_p90": percentile(match_maes, 90)}
    return out


# ------------------------------------------------------------------------------------------------------
# one match, three conditions, per-tick rows at STEP_TICKS cadence
# ------------------------------------------------------------------------------------------------------
def run_match_audit(env, model, deck, db: CardDB, entry: dict, k: int, cfg: dict,
                    warned_costless: Optional[set] = None, whitelist: Optional[frozenset] = None,
                    charge_trace: bool = False) -> tuple[list[dict], dict, list[dict], list[dict]]:
    """Mirrors e1_eval.run_match's while-loop (same live policy, same accepted-play bookkeeping, IDENTICAL
    decisions at IDENTICAL ticks) but steps the engine STEP_TICKS at a time, updates all four estimators
    every step, and records a per-tick truth/estimate row instead of only a match summary. Returns
    (ticks, match_summary, charge_rows, track_event_rows) -- the last two are always [] unless
    ``charge_trace`` is set, in which case condition A's estimator becomes a ``TracedEstimator`` (module
    docstring O10) and they carry this match's charge / track-expiry trace rows; every other condition
    (A+, A_wl, B) is untouched regardless of this flag. ``whitelist``: FIX 4's ``load_detector_cards()``
    result, required for A_wl."""
    if warned_costless is None:
        warned_costless = set()
    if whitelist is None:
        whitelist = frozenset()
    step = int(cfg.get("estimator_step") or STEP_TICKS)
    decide_every = int(cfg["decide_every"])
    if decide_every % step != 0:
        raise SystemExit(f"--decide-every {decide_every} must be a multiple of --estimator-step {step}")

    t0 = time.perf_counter()
    state = env.reset(entry)
    side, mirror = env.side, env._mirror
    engine_deck, deck_index_of_slot, costs = _slot_maps(env, deck, entry)
    tag = str(entry["tag"])
    rng_obs = np.random.default_rng(obs_seed(tag, k))          # e1_eval-identical stream -- POLICY ticks only
    rng_obs_b5 = np.random.default_rng(b5_seed(tag, k))        # harness-local -- NON-policy 5-tick steps only
    policy, grid, device = cfg["policy"], cfg["grid"], cfg["device"]
    unmapped: set = set()
    done_plays: list[tuple[int, int, float, float]] = []
    last_play_tick: Optional[int] = None
    first_tick: Optional[int] = None
    est_ap = TrackedEstimator(db)      # A+  : units + spells, engine truth, no whitelist
    # O10: condition A ONLY becomes a TracedEstimator when --charge-trace is set; TrackedEstimator otherwise,
    # byte-identical to before the flag existed (TracedEstimator IS-A TrackedEstimator, so est_a._est/
    # est_a.charged_by_base are unaffected either way -- only the extra last_charges/last_expired bookkeeping
    # is new, and nothing reads it unless charge_trace is True).
    est_a = TracedEstimator(db) if charge_trace else TrackedEstimator(db)   # A: units only, no whitelist
    est_wl = TrackedEstimator(db)      # A_wl: units only, engine truth, live's detector_cards whitelist (FIX 4)
    est_b = TrackedEstimator(db)       # B   : units only, degraded live-like view
    inited = False
    prev_truth: Optional[float] = None
    ticks: list[dict] = []
    charge_rows: list[dict] = []       # O10: condition A's charge trace (only ever populated if charge_trace)
    track_event_rows: list[dict] = []  # O10: condition A's track-expiry trace (ditto)
    prev_a_dets: Optional[list] = None # O10: PREVIOUS sample's condition-A det list, for charge_diagnostics
    prev_a_t: Optional[float] = None   # O10: PREVIOUS sample's t_sec, for dt_since_prev_sample_s
    t_to_tick: dict = {}                # O10: this match's t_sec -> tick map, for track_events' last_seen_tick
    n_dec = n_acc = 0
    max_ticks = int(cfg.get("max_ticks") or 0)
    done = False
    while not done:
        tick = int(env.tick)
        if first_tick is None:
            first_tick = tick
        if last_play_tick is None:
            last_play_tick = tick
        pol_tick = is_policy_tick(tick, first_tick, decide_every)

        bs = from_engine(ep.compact_raw(state), side, deck, engine_deck=engine_deck, unmapped=unmapped)
        # Condition B's view: the REAL e1_eval stream on policy ticks (so the policy decision below is
        # byte-identical to attempt 1 / e1_eval), the harness-local stream on the 5-tick-only steps in
        # between (CHANGE 1 -- e1_eval has no cadence of its own for those).
        view = live_view(bs, rng_obs if pol_tick else rng_obs_b5, deck, Noise())
        if not inited:
            est_ap.reset(my_elixir=bs.my_elixir, now=bs.t_sec)
            est_a.reset(my_elixir=bs.my_elixir, now=bs.t_sec)
            est_wl.reset(my_elixir=bs.my_elixir, now=bs.t_sec)
            est_b.reset(my_elixir=view.my_elixir, now=bs.t_sec)
            inited = True

        truth = float(bs.opp_elixir)
        # FIX 2: is_opp_play_tick is filled in AFTER the loop (opp_play_flags, from env.ghost_events) --
        # truth_drop stays as an informational column only, it no longer decides the flag.
        truth_drop = 0.0 if prev_truth is None else (prev_truth - truth)

        # update() RETURNS the estimate normalized to [0, 1] (opponent_elixir.py:131); the 0-10 elixir-unit
        # value this harness grades is ``._est`` (set as a side effect of the same call, opponent_elixir.py
        # :130) -- attempt 1's section-1.9 near-miss. Call update() for its side effect, then read ``._est``.
        ap_dets = dets_of(bs.units) + dets_of_costed(bs.spells, db, warned_costless)
        est_ap.update(bs.my_elixir, ap_dets, bs.t_sec)
        est_ap_val = float(est_ap._est)
        n_ap = n_enemy(bs.units) + n_enemy(bs.spells)

        a_dets = dets_of(bs.units)
        if charge_trace:
            ch_rows, tr_rows = run_traced_update(est_a, bs.my_elixir, a_dets, bs.t_sec, tick, tag, k,
                                                 prev_a_dets, prev_a_t, t_to_tick)
            charge_rows.extend(ch_rows)
            track_event_rows.extend(tr_rows)
            prev_a_dets, prev_a_t = a_dets, bs.t_sec
        else:
            est_a.update(bs.my_elixir, a_dets, bs.t_sec)
        est_a_val = float(est_a._est)
        n_a = n_enemy(bs.units)

        # FIX 4: A_wl -- perfect detection, units only, WITH live's detector_cards whitelist applied.
        wl_dets = dets_of_whitelisted(bs.units, whitelist)
        est_wl.update(bs.my_elixir, wl_dets, bs.t_sec)
        est_wl_val = float(est_wl._est)
        n_wl = sum(1 for d in wl_dets if d.team == "enemy")

        est_b.update(view.my_elixir, dets_of(view.units), bs.t_sec)
        est_b_val = float(est_b._est)
        n_b = n_enemy(view.units)

        # --- our own decision (same live rule as e1_eval.run_match), POLICY ticks only, unchanged cadence ---
        is_my_play_tick = False
        if pol_tick:
            tok, mask, sc = to_tokens(view, MAX_U)
            past = _past(done_plays, tick)
            enc, heads, p, hand = model_forward(model, tok, mask, sc, past, device)
            n_dec += 1
            el_int = float(int(view.my_elixir))
            allowed = allowed_slots(hand, costs, el_int, afford_mask=cfg["afford_mask"])
            st = anti_stall(el_int, tick, last_play_tick, cfg["stall_elixir"], cfg["stall_seconds"])
            d = live_decide(model, enc, heads, p, allowed, tau=cfg["tau"], stalled=st, device=device)
            if d["play"]:
                x, y = ep.cell_center(d["cell"], grid)
                X, Y = ep.cell_to_engine(d["cell"], mirror, grid)
                r = env.eng.act(side=side, deck_index=deck_index_of_slot[d["slot"]], x=X, y=Y)
                if bool(r["accepted"]):
                    n_acc += 1
                    done_plays.append((tick, d["slot"], x, y))
                    last_play_tick = tick
                    base = vocab.base_key(deck.cards[d["slot"]])
                    est_ap.record_my_play(base)
                    est_a.record_my_play(base)
                    est_wl.record_my_play(base)
                    est_b.record_my_play(base)
                    is_my_play_tick = True

        ticks.append({"tag": tag, "k": int(k), "tick": tick, "t_sec": round(bs.t_sec, 3), "phase": phase_of(bs),
                     "truth": truth, "truth_drop": round(truth_drop, 4),
                     "est_Aplus": round(est_ap_val, 4), "est_A": round(est_a_val, 4),
                     "est_Awl": round(est_wl_val, 4), "est_B": round(est_b_val, 4),
                     "n_enemy_dets_Aplus": n_ap, "n_enemy_dets_A": n_a, "n_enemy_dets_Awl": n_wl,
                     "n_enemy_dets_B": n_b,
                     "is_opp_play_tick": False,        # FIX 2: filled in below from env.ghost_events, not here
                     "is_my_play_tick": bool(is_my_play_tick), "is_policy_tick": bool(pol_tick)})
        prev_truth = truth
        if max_ticks and len(ticks) >= max_ticks:
            break

        env._advance_to(min(env.tick + step, env.tail_cap))
        state = env.eng.observe()
        done = bool(env.terminated) or env.tick >= env.tail_cap

    # FIX 2: is_opp_play_tick from the ghost-drive path's own delivery log, not the truth-drop threshold
    # (which silently misses every 1-cost play -- module docstring). env.ghost_events is reset to [] per
    # match by engine_env.py's reset(), so by here it holds exactly THIS match's (tick, accepted, reason) log.
    flags = opp_play_flags([row["tick"] for row in ticks], list(env.ghost_events))
    for row, flag in zip(ticks, flags):
        row["is_opp_play_tick"] = bool(flag)

    outcome, crowns = ep._outcome(env, state)
    # FIX 1: the over-charge TRUTH ledger is what was actually DELIVERED (env.ghost_cards_delivered), never
    # the script (ghost_scripted_by_base is kept below for reference only, clearly labelled, unused in math).
    ghost_scripted = ghost_scripted_by_base(entry, db, warned_costless)
    ghost_delivered = ghost_delivered_by_base(dict(env.ghost_cards_delivered), db, warned_costless)
    ghost_delivered_total = sum(v[1] for v in ghost_delivered.values())
    charged_by_base = {"Aplus": dict(est_ap.charged_by_base), "A": dict(est_a.charged_by_base),
                       "A_wl": dict(est_wl.charged_by_base), "B": dict(est_b.charged_by_base)}
    over_charge_at_end = {cond: round(sum(v[1] for v in d.values()) - ghost_delivered_total, 2)
                          for cond, d in charged_by_base.items()}
    summary = {"tag": tag, "k": int(k), "n_ticks": len(ticks), "decisions": n_dec, "plays_accepted": n_acc,
              "outcome": outcome, "crowns_for": int(crowns[0]), "crowns_against": int(crowns[1]),
              "end_tick": int(env.tick), "unmapped": sorted(unmapped),
              "wall_s": round(time.perf_counter() - t0, 1),
              "charged_by_base": charged_by_base,
              "ghost_scripted_by_base": ghost_scripted,           # reference only -- NOT used for over-charge
              "ghost_delivered_by_base": ghost_delivered,         # the truth ledger over-charge actually uses
              "over_charge_at_end": over_charge_at_end}
    return ticks, summary, charge_rows, track_event_rows


# ------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ckpt", type=Path, required=True, help="S1 checkpoint the 'live' policy drives")
    ap.add_argument("--pool", type=Path, default=POOL_V1)
    ap.add_argument("--split-file", type=Path, default=None)
    ap.add_argument("--split", choices=("heldout", "train"), default="heldout")
    ap.add_argument("--entries", default="all", help="a:b over the split's entries in pool order, or 'all'")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--tau", type=float, default=TAU_LIVE)
    ap.add_argument("--no-afford-mask", action="store_true")
    ap.add_argument("--stall-elixir", default=str(STALL_ELIXIR_LIVE), help="'none' disables anti-stall")
    ap.add_argument("--stall-seconds", type=float, default=STALL_SECONDS_LIVE)
    ap.add_argument("--decide-every", type=int, default=DECIDE_EVERY, help="policy decision cadence, ticks")
    ap.add_argument("--estimator-step", type=int, default=STEP_TICKS,
                    help="CHANGE 1: estimator update() cadence, ticks; must divide --decide-every")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=20.0, help="engine socket timeout (s) -- kept short so a "
                    "dead port fails cleanly instead of hanging on the 120s EngineMatchEnv default")
    ap.add_argument("--max-ticks", type=int, default=0, help="stop after this many TOTAL per-tick rows across "
                    "the run (smoke); 0 = unlimited. SMOKE ONLY: truncates each match mid-flight, then "
                    "_outcome() is computed on a NON-terminal engine state (outcome/crowns are meaningless "
                    "for a truncated match; ticks/estimates up to the cut are still valid)")
    ap.add_argument("--charge-trace", action="store_true", help="O10: for condition A ONLY, write "
                    "<out>/charges.jsonl (one row per _opp_spent charge, with diagnostics + a best-effort "
                    "reason) and <out>/track_events.jsonl (one row per track expiry). No fix -- "
                    "instrumentation only; off by default and produces no new files / no summary change.")
    ap.add_argument("--out", type=Path, required=True)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    refuse_existing_out(a.out, resume=False)
    shard = parse_shard(a.shard)
    seeds = parse_seeds(a.seeds)
    stall_elixir = None if str(a.stall_elixir).lower() == "none" else float(a.stall_elixir)
    if int(a.decide_every) % int(a.estimator_step) != 0:
        raise SystemExit(f"--decide-every {a.decide_every} must be a multiple of --estimator-step {a.estimator_step}")
    import torch
    torch.set_num_threads(max(1, int(a.threads)))

    pool_path = Path(a.pool)
    split_path = Path(a.split_file) if a.split_file else pool_path.with_name(pool_path.stem + "_split.json")
    frozen = json.loads(split_path.read_text(encoding="utf-8"))
    pool_sha = sha256_file(pool_path)
    if pool_sha != frozen["pool_sha256"]:
        raise SystemExit(f"REFUSING: pool sha256 {pool_sha} != frozen split's {frozen['pool_sha256']}")
    rows = load_pool_v1(pool_path)
    entries = select_split(rows, a.split)
    del rows
    idx = parse_entries(a.entries, entries)
    tasks = make_tasks(idx, seeds, shard)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    deck = load_deck("icebow")
    db = CardDB(path=deck.config.parent / "cards.yaml")
    whitelist = load_detector_cards(deck.config)          # FIX 4: A_wl's live detector_cards whitelist
    model, minfo = ep.load_model(a.ckpt, a.device)
    slot = SLOT_OF_PORT.get(int(a.port), int(a.port))
    cfg = {"policy": "live", "tau": float(a.tau), "afford_mask": not a.no_afford_mask, "stall_elixir": stall_elixir,
          "stall_seconds": float(a.stall_seconds), "grid": minfo.get("grid", "floor"), "device": a.device,
          "decide_every": int(a.decide_every), "estimator_step": int(a.estimator_step), "max_ticks": int(a.max_ticks)}
    print(json.dumps({"opp_est_audit": True, "port": a.port, "tasks": len(tasks), "grid": cfg["grid"],
                      "tau": cfg["tau"], "decide_every": cfg["decide_every"],
                      "estimator_step": cfg["estimator_step"], "n_detector_cards": len(whitelist)}), flush=True)

    tick_path = out / "ticks.jsonl"
    charge_trace = bool(a.charge_trace)
    # O10: charges.jsonl / track_events.jsonl are ONLY ever created when --charge-trace is passed -- off by
    # default means no new files at all, per the ticket.
    charges_fh = (out / "charges.jsonl").open("a", encoding="utf-8") if charge_trace else None
    track_events_fh = (out / "track_events.jsonl").open("a", encoding="utf-8") if charge_trace else None
    all_rows: list[dict] = []
    match_summaries: list[dict] = []
    warned_costless: set = set()
    env = None
    total_ticks = 0
    try:
        try:
            env = PoolV1Env(port=int(a.port), host=a.host, decision_ticks=int(a.decide_every), timeout=float(a.timeout))
        except Exception as exc:
            print(f"[opp_est_audit] ENGINE UNREACHABLE on {a.host}:{a.port}: {exc!r}", flush=True)
            return 2
        with tick_path.open("a", encoding="utf-8") as fh:
            for (i, k) in tasks:
                entry = entries[i]
                try:
                    match_rows, msummary, ch_rows, tr_rows = run_match_audit(
                        env, model, deck, db, entry, k, cfg, warned_costless, whitelist,
                        charge_trace=charge_trace)
                except Exception as exc:
                    print(f"[opp_est_audit] ERROR on {entry['tag']} k={k}: {exc!r}", flush=True)
                    return 3
                for row in match_rows:
                    fh.write(json.dumps(row) + "\n")
                fh.flush()
                if charge_trace:
                    for row in ch_rows:
                        charges_fh.write(json.dumps(row) + "\n")
                    for row in tr_rows:
                        track_events_fh.write(json.dumps(row) + "\n")
                    charges_fh.flush()
                    track_events_fh.flush()
                all_rows.extend(match_rows)
                match_summaries.append(msummary)
                total_ticks += len(match_rows)
                print(f"[opp_est_audit] {entry['tag']} k={k} ticks={len(match_rows)} "
                     f"accepted={msummary['plays_accepted']} {msummary['wall_s']}s", flush=True)
                if a.max_ticks and total_ticks >= a.max_ticks:
                    break
    finally:
        if env is not None:
            env.close()
        if charges_fh is not None:
            charges_fh.close()
        if track_events_fh is not None:
            track_events_fh.close()

    conds = (("Aplus_perfect_detection_with_spells", "est_Aplus", "Aplus"),
             ("A_perfect_detection", "est_A", "A"), ("A_wl_live_whitelist", "est_Awl", "A_wl"),
             ("B_degraded_live", "est_B", "B"))
    # FIX 1: ghost_delivered_run is THE truth ledger fed to every overcharge_table below; ghost_scripted_run
    # is carried in summary.json for reference only and is never used in an over-charge computation.
    ghost_delivered_run = merge_base_ledgers([m["ghost_delivered_by_base"] for m in match_summaries])
    ghost_scripted_run = merge_base_ledgers([m["ghost_scripted_by_base"] for m in match_summaries])
    # FIX 3: per-match record, field-for-field the same spelling e1_eval.run_match writes to matches.jsonl
    # (outcome, crowns_for, crowns_against, plays_accepted) -- the lead's fidelity gate against ctrl_live100.
    matches = [{"tag": m["tag"], "k": m["k"], "outcome": m["outcome"], "crowns_for": m["crowns_for"],
               "crowns_against": m["crowns_against"], "plays_accepted": m["plays_accepted"],
               "decisions": m["decisions"], "end_tick": m["end_tick"], "n_ticks": m["n_ticks"]}
              for m in match_summaries]
    summary = {
        "n_ticks": len(all_rows), "n_matches": len(match_summaries), "port": a.port, "split": a.split,
        "entries": a.entries, "seeds": a.seeds, "ckpt": str(a.ckpt),
        "decide_every": cfg["decide_every"], "estimator_step": cfg["estimator_step"],
        "n_detector_cards": len(whitelist),
        "matches": matches,
        "plays_accepted_by_match": {f"{m['tag']}:{m['k']}": m["plays_accepted"] for m in match_summaries},
        "over_charge_at_end_by_match": [{"tag": m["tag"], "k": m["k"], **m["over_charge_at_end"]}
                                        for m in match_summaries],
        "ghost_delivered_by_base": ghost_delivered_run,             # the truth ledger over-charge uses
        "ghost_scripted_by_base": ghost_scripted_run,                # REFERENCE ONLY -- not used for over-charge
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for key, est_key, cond in conds:
        summary[key] = summarize_condition(all_rows, est_key)
        charged_run = merge_base_ledgers([m["charged_by_base"][cond] for m in match_summaries])
        summary[key]["overcharge"] = overcharge_table(charged_run, ghost_delivered_run)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    md = ["# opp_est_audit summary", "", f"n_ticks={summary['n_ticks']} n_matches={summary['n_matches']} "
         f"port={a.port} split={a.split} entries={a.entries} decide_every={cfg['decide_every']} "
         f"estimator_step={cfg['estimator_step']}", "",
         "| condition | MAE | bias | P90\\|err\\| | share<=1.0 | share<=2.0 |",
         "|---|---|---|---|---|---|"]
    labels = {"Aplus_perfect_detection_with_spells": "A+ perfect detection (+spells, no whitelist)",
             "A_perfect_detection": "A perfect detection (units only, NO whitelist -- loose upper bound)",
             "A_wl_live_whitelist": "A_wl perfect detection (units only, live's detector_cards whitelist)",
             "B_degraded_live": "B degraded (live)"}
    for key, _, _cond in conds:
        m = summary[key]
        md.append(f"| {labels[key]} | {m['mae']} | {m['mean_bias']} | {m['p90_abs_err']} | "
                 f"{m['share_abs_err_le_1_0']} | {m['share_abs_err_le_2_0']} |")
    md.append("")
    for key, _, _cond in conds:
        md.append(f"## Condition {labels[key]}")
        for ph in PHASES:
            p = summary[key]["by_phase"][ph]
            md.append(f"- {ph}: n={p['n']} MAE={p['mae']} bias={p['mean_bias']} P90={p['p90_abs_err']}")
        op, mp = summary[key]["opp_play_ticks"], summary[key]["my_play_ticks"]
        md.append(f"- opponent-play ticks: n={op['n']} MAE={op['mae']} bias={op['bias']}")
        md.append(f"- our accepted-play ticks: n={mp['n']} MAE={mp['mae']} bias={mp['bias']}")
        nc, bc = summary[key]["by_truth_cap"]["near_cap_ge_9"], summary[key]["by_truth_cap"]["below_cap"]
        md.append(f"- truth>=9.0 (near cap): n={nc['n']} MAE={nc['mae']} bias={nc['mean_bias']}")
        md.append(f"- truth<9.0: n={bc['n']} MAE={bc['mae']} bias={bc['mean_bias']}")
        bm = summary[key]["by_match"]
        md.append(f"- per-match MAE: n_matches={bm['n_matches']} median={bm['mae_median']} p90={bm['mae_p90']}")
        oc = summary[key]["overcharge"]
        md.append(f"- OVER-CHARGE vs DELIVERED (FIX 1, not scripted): total_charged={oc['total_charged_elixir']} "
                 f"total_ghost_delivered={oc['total_ghost_elixir']} total_over_charge={oc['total_over_charge']} "
                 f"share_from_never_played_bases={oc['over_charge_share_from_bases_never_played']}")
        md.append("  | base | ghost_plays(delivered) | ghost_elixir | charges | charged_elixir | over_charge |")
        md.append("  |---|---|---|---|---|---|")
        for row in oc["top"][:10]:
            md.append(f"  | {row['base']} | {row['ghost_plays']} | {row['ghost_elixir']} | "
                     f"{row['charges']} | {row['charged_elixir']} | {row['over_charge']} |")
        md.append("  (top 10 shown here; top 25 in summary.json)")
        md.append("")
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"OPP_EST_AUDIT_DONE": {
        "n_ticks": summary["n_ticks"], "n_matches": summary["n_matches"],
        "matches": summary["matches"],                            # FIX 3: fidelity gate vs ctrl_live100
        "plays_accepted_by_match": summary["plays_accepted_by_match"],
        "Aplus_mae": summary["Aplus_perfect_detection_with_spells"]["mae"],
        "A_mae": summary["A_perfect_detection"]["mae"], "A_bias": summary["A_perfect_detection"]["mean_bias"],
        "A_wl_mae": summary["A_wl_live_whitelist"]["mae"], "A_wl_bias": summary["A_wl_live_whitelist"]["mean_bias"],
        "B_mae": summary["B_degraded_live"]["mae"], "B_bias": summary["B_degraded_live"]["mean_bias"],
        "A_wl_total_over_charge": summary["A_wl_live_whitelist"]["overcharge"]["total_over_charge"],
        "A_wl_over_charge_share_never_played": summary["A_wl_live_whitelist"]["overcharge"][
            "over_charge_share_from_bases_never_played"],
        "A_wl_over_charge_top3": summary["A_wl_live_whitelist"]["overcharge"]["top"][:3],
    }}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
