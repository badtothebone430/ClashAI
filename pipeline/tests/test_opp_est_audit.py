"""Offline tests for pipeline/opp_est_audit.py (O8).

    icebow/.venv/Scripts/python.exe -m unittest pipeline.tests.test_opp_est_audit -v

No engine, no GPU. Covers: (i) dets conversion -- every enemy class in the held-out pool's ghost decks
resolves to a non-None CardDB cost (skipped, not silently, if the pool file is absent); (ii) phase bucketing
against obs_contract._phase; (iii) the metric functions on a hand-built toy series with known MAE/bias/P90;
(iv) a 3-tick estimator plumbing check (reset -> update with one enemy 4-cost det -> estimate drops by
exactly the cost, per the estimator's own exact-books semantics -- opponent_elixir.py:113); (v, Attempt 2
CHANGE 1) the 5-tick stepping loop calls the live policy exactly on the original 10-tick decision boundaries;
(vi, Attempt 2 CHANGE 2) the A+ dets path resolves a known spell cost (fireball=4) and skips + warns-once on
a costless one (mirror -- dynamic pricing, found in Attempt 1's dets-conversion test); (vii, Attempt 3) a
synthetic single 4-cost enemy det makes ``TrackedEstimator.charged_by_base == {base: [1, 4.0]}`` (plus a
two-different-bases-in-one-update() case, the scenario a before/after ``_opp_spent`` diff cannot attribute);
(viii, Attempt 3) ``ghost_scripted_by_base`` built correctly from a hand-built ``ghost_commands`` list
(ability entries excluded, hyphenated slugs normalized, a costless slug skipped + warned once); (ix, Attempt
4 FIX 1) ``ghost_delivered_by_base`` (the truth ledger over-charge actually uses) differs from
``ghost_scripted_by_base`` when a scripted play was never delivered; (x, Attempt 4 FIX 2) ``opp_play_flags``
flags only the first sample tick at/after a DELIVERED ghost event; (xi, Attempt 4 FIX 4) ``dets_of_whitelisted``
drops a non-whitelisted base and keeps a whitelisted one, and ``load_detector_cards`` loads the real,
non-empty live whitelist; (xii, Attempt 4 FIX 5) hand-written (not self-referential) phase expectations, plus
``overcharge_table``/``merge_base_ledgers`` on hand-built ledgers.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_ICEBOW_SRC = REPO / "icebow" / "src"
if str(_ICEBOW_SRC) not in sys.path:
    sys.path.insert(0, str(_ICEBOW_SRC))

from pipeline import vocab                                            # noqa: E402
from pipeline.e1_pool import POOL_V1, load_pool_v1, select_split      # noqa: E402
from pipeline.obs_contract import Unit, _phase, load_deck             # noqa: E402
from pipeline.opp_est_audit import (                                  # noqa: E402
    TrackedEstimator, TracedEstimator, _Det, b5_seed, card_cost, charge_diagnostics, classify_charge_reason,
    dets_of_costed, dets_of_whitelisted, ghost_delivered_by_base, ghost_scripted_by_base, is_policy_tick,
    load_detector_cards, mae, mean_bias, merge_base_ledgers, opp_play_flags, overcharge_table, percentile,
    phase_from_flags, phase_of, run_traced_update, share_within, summarize_condition,
)

from clashrl.cards import CardDB                                      # noqa: E402
from clashrl.opponent_elixir import OpponentElixirEstimator           # noqa: E402


def _db() -> CardDB:
    deck = load_deck("icebow")
    return CardDB(path=deck.config.parent / "cards.yaml")


# ------------------------------------------------------------------------------------------------------
# (i) dets conversion: every enemy class the held-out pool's ghost decks actually deploy must resolve a cost
# ------------------------------------------------------------------------------------------------------
class TestDetsConversion(unittest.TestCase):
    def test_known_card_resolves(self):
        db = _db()
        for base in ("musketeer", "knight", "giant", "skeletons"):
            self.assertIsNotNone(db.elixir(base), f"{base} must have a known elixir cost")

    def test_heldout_ghost_decks_resolve_cost(self):
        """Every BASE card any held-out ghost deck carries must price through CardDB.elixir. This is the
        exact path a real enemy play takes: engine name -> vocab.engine_unit_id -> vocab.UNIT_VOCAB ->
        vocab.base_key -> CardDB.elixir (opponent_elixir.py:109's ``self.db.elixir(base)``)."""
        if not POOL_V1.exists():
            self.skipTest(f"pool file missing: {POOL_V1}")
        db = _db()
        pool = load_pool_v1(POOL_V1)
        heldout = select_split(pool, "heldout")
        self.assertGreater(len(heldout), 0, "held-out split must be non-empty for this test to mean anything")
        names: set[str] = set()
        for e in heldout:
            for it in e.get("ghost_deck") or []:
                names.add(str(it["name"]))
        self.assertGreater(len(names), 0)
        unresolved: list[tuple[str, str]] = []       # (engine name, base key) whose cost came back None
        unmapped: list[str] = []                      # engine name with no vocab id at all
        for name in sorted(names):
            cid = vocab.engine_unit_id(name)
            if cid is None:
                unmapped.append(name)
                continue
            base = vocab.base_key(vocab.UNIT_VOCAB[cid])
            if db.elixir(base) is None:
                unresolved.append((name, base))
        # FINDING (not silently skipped): print so -v output and the harness report both carry it.
        if unmapped:
            print(f"[test_dets_conversion] {len(unmapped)} ghost-deck name(s) have NO vocab id: {unmapped}")
        if unresolved:
            print(f"[test_dets_conversion] {len(unresolved)} base key(s) resolve to NO CardDB cost: "
                 f"{unresolved}")
        # The conversion pipeline itself (name -> vocab id) must always succeed for a card that made it into
        # a deck (decks are built from the live catalog, which vocab.py is derived from) -- THIS is a hard
        # assertion. Whether every resulting base then has a priced cards.yaml row is reported above as a
        # finding, not asserted, per the ticket's "list any that do NOT ... do not silently skip".
        self.assertEqual(unmapped, [], f"deck card(s) with no vocab id at all: {unmapped}")


# ------------------------------------------------------------------------------------------------------
# (ii) phase bucketing against obs_contract._phase
# ------------------------------------------------------------------------------------------------------
class TestPhaseBucketing(unittest.TestCase):
    def test_matches_phase_helper_at_hand_written_times(self):
        """FIX 5: expectations written by hand from the DOCUMENTED thresholds (obs_contract.py:
        DOUBLE_ELIXIR_S=120.0, OVERTIME_S=180.0), never by re-deriving them through the same if-chain
        ``phase_from_flags`` itself uses -- the old version of this test was self-referential (it built
        ``expect`` with an identical elif ladder) and could not have caught a bug in that ladder."""
        cases = {30.0: "single", 150.0: "double", 200.0: "overtime"}
        for t, expect in cases.items():
            double, overtime = _phase(t)          # obs_contract's own thresholds, independent computation
            self.assertEqual(phase_from_flags(double, overtime), expect, f"t={t}")
            # and directly against the hand-written thresholds too, with no dependency on _phase at all:
            hand_double, hand_overtime = (t >= 120.0), (t >= 180.0)
            self.assertEqual(phase_from_flags(hand_double, hand_overtime), expect, f"t={t}")

    def test_boundaries(self):
        self.assertEqual(phase_from_flags(False, False), "single")
        self.assertEqual(phase_from_flags(True, False), "double")
        self.assertEqual(phase_from_flags(True, True), "overtime")
        # overtime implies double_elixir is also True in obs_contract._phase (180 >= 120); phase_of trusts
        # the caller's flags directly, so an inconsistent (False, True) still reports "overtime" -- that is
        # by design (overtime always wins), tested explicitly since it is the one flag combo _phase never
        # actually produces.
        self.assertEqual(phase_from_flags(False, True), "overtime")


# ------------------------------------------------------------------------------------------------------
# (iii) metric functions on a hand-built toy series with known MAE / bias / P90
# ------------------------------------------------------------------------------------------------------
class TestMetrics(unittest.TestCase):
    def test_mae_bias(self):
        # est - truth: -1, +2, -3, +4, -5 -> abs = 1,2,3,4,5 mean 3.0; signed mean = (-1+2-3+4-5)/5 = -0.6
        errs = [-1.0, 2.0, -3.0, 4.0, -5.0]
        self.assertAlmostEqual(mae(errs), 3.0)
        self.assertAlmostEqual(mean_bias(errs), -0.6)

    def test_percentile_linear_interp(self):
        # sorted [1,2,3,4,5,6,7,8,9,10]; p90 rank = 0.9*9 = 8.1 -> between index 8 (9) and 9 (10), frac .1
        # -> 9 + 0.1*(10-9) = 9.1 (numpy.percentile([1..10], 90) == 9.1, verified by hand: linear method)
        vals = list(range(1, 11))
        self.assertAlmostEqual(percentile(vals, 90), 9.1)
        self.assertAlmostEqual(percentile([5.0], 90), 5.0)
        self.assertIsNone(percentile([], 90))

    def test_share_within(self):
        abs_errs = [0.2, 0.9, 1.0, 1.5, 2.0, 3.0]
        self.assertAlmostEqual(share_within(abs_errs, 1.0), 3 / 6)   # 0.2, 0.9, 1.0
        self.assertAlmostEqual(share_within(abs_errs, 2.0), 5 / 6)   # all but 3.0

    def test_summarize_condition_toy(self):
        rows = [
            {"tag": "m1", "k": 0, "phase": "single", "truth": 5.0, "est_A": 6.0,
             "is_opp_play_tick": False, "is_my_play_tick": False},
            {"tag": "m1", "k": 0, "phase": "double", "truth": 4.0, "est_A": 2.0,
             "is_opp_play_tick": True, "is_my_play_tick": False},
            {"tag": "m2", "k": 0, "phase": "single", "truth": 3.0, "est_A": 3.0,
             "is_opp_play_tick": False, "is_my_play_tick": True},
        ]
        s = summarize_condition(rows, "est_A")
        self.assertEqual(s["n_ticks"], 3)
        # errs: +1, -2, 0 -> mae = 1.0, bias = -1/3
        self.assertAlmostEqual(s["mae"], 1.0)
        self.assertAlmostEqual(s["mean_bias"], -1.0 / 3.0)
        self.assertEqual(s["by_phase"]["single"]["n"], 2)
        self.assertEqual(s["by_phase"]["double"]["n"], 1)
        self.assertAlmostEqual(s["by_phase"]["double"]["mae"], 2.0)
        self.assertEqual(s["opp_play_ticks"]["n"], 1)
        self.assertAlmostEqual(s["opp_play_ticks"]["mae"], 2.0)
        self.assertEqual(s["my_play_ticks"]["n"], 1)
        self.assertAlmostEqual(s["my_play_ticks"]["mae"], 0.0)
        self.assertEqual(s["by_match"]["n_matches"], 2)


# ------------------------------------------------------------------------------------------------------
# (iv) 3-tick estimator plumbing check: reset -> update with one enemy 4-cost det -> estimate drops by cost
# ------------------------------------------------------------------------------------------------------
class TestEstimatorPlumbing(unittest.TestCase):
    def test_reset_then_one_enemy_play_drops_estimate_by_exact_cost(self):
        db = _db()
        cost = db.elixir("musketeer")
        self.assertEqual(cost, 4, "musketeer must be a 4-cost card for this to be the ticket's '4-cost det'")
        est = OpponentElixirEstimator(db)

        # tick 1: reset at my_elixir=5.0, t=0
        est.reset(my_elixir=5.0, now=0.0)
        self.assertAlmostEqual(est._est, 5.0)

        # tick 2: no enemy dets, my_elixir regenerated to 5.7 -- the estimator has NO clock of its own (it
        # never reads `now` to regenerate; `now` only ages tracks for the forget/cluster window), so the
        # estimate tracks the CALLER's my_elixir exactly when nothing has been spent by either side.
        v2 = est.update(5.7, [], 1.0)
        self.assertAlmostEqual(v2 * 10.0, 5.7, places=4)
        self.assertAlmostEqual(est._est, 5.7)

        # tick 3: one fresh enemy musketeer (4 cost) appears; my_elixir has regenerated further to 6.0.
        # est = my_elixir + my_spent(0) - opp_spent(cost) EXACTLY (opponent_elixir.py:113), no clipping
        # triggers here (6.0 - 4 = 2.0 is inside [0, 10]) so the drop is exact, not merely bounded.
        det = _Det(base="musketeer", cx=0.5, gy=0.3, team="enemy")
        v3 = est.update(6.0, [det], 2.0)
        self.assertAlmostEqual(est._est, 6.0 - cost, places=4)
        self.assertAlmostEqual(v3 * 10.0, 6.0 - cost, places=4)
        # direction + bound, independent of the exact-arithmetic assertion above:
        self.assertLess(est._est, 6.0)
        self.assertGreaterEqual(est._est, 6.0 - cost - 1e-6)

    def test_dets_of_units_shape(self):
        """card_cost() + the _Det shim round-trip a vocab id the way run_match_audit builds dets."""
        db = _db()
        cid = vocab.unit_id("musketeer")
        self.assertEqual(card_cost(db, cid), 4)


# ------------------------------------------------------------------------------------------------------
# (v) Attempt 2 CHANGE 1: the 5-tick stepping loop must call the live policy exactly on the ORIGINAL
# 10-tick decision boundaries (identical ticks to Attempt 1 / e1_eval), never on the in-between 5-tick steps.
# ------------------------------------------------------------------------------------------------------
class TestFiveTickCadence(unittest.TestCase):
    def test_policy_fires_only_on_original_decision_ticks(self):
        first_tick, decide_every, step = 90, 10, 5
        ticks = [first_tick + i * step for i in range(40)]        # 90, 95, 100, 105, ... the run_match_audit walk
        got = [is_policy_tick(t, first_tick, decide_every) for t in ticks]
        want = [i % 2 == 0 for i in range(40)]                    # every OTHER 5-tick step, starting True at i=0
        self.assertEqual(got, want)
        policy_ticks = [t for t, flag in zip(ticks, got) if flag]
        self.assertEqual(policy_ticks, list(range(first_tick, first_tick + 40 * step, decide_every)))
        # every policy tick agrees with the direct definition e1_eval itself would decide on
        for t in ticks:
            self.assertEqual(is_policy_tick(t, first_tick, decide_every), (t - first_tick) % decide_every == 0)

    def test_b5_seed_differs_from_obs_seed_domain(self):
        """The harness-local stream (non-policy 5-tick steps, Condition B only) must never collide with
        e1_eval's own obs_seed domain -- same crc32 shape, different domain string."""
        from pipeline.e1_eval import obs_seed
        for tag, k in (("000YLL9RURY8", 0), ("abc", 3)):
            self.assertNotEqual(b5_seed(tag, k), obs_seed(tag, k))


# ------------------------------------------------------------------------------------------------------
# (vi) Attempt 2 CHANGE 2: A+ dets path -- a known spell cost resolves, a costless one is skipped and
# warned exactly once (never silently, never charged as free).
# ------------------------------------------------------------------------------------------------------
def _spell_unit(base: str, side: int = 1) -> Unit:
    return Unit(cls=vocab.unit_id(base), side=side, x=0.5, y=0.3, hp_frac=None, deploying=None,
               age_sec=None, conf=1.0)


class TestAplusSpellPath(unittest.TestCase):
    def test_fireball_resolves_cost_four(self):
        db = _db()
        self.assertEqual(db.elixir("fireball"), 4, "fireball must be a 4-cost card for this check")
        warned: set = set()
        dets = dets_of_costed([_spell_unit("fireball")], db, warned)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].base, "fireball")
        self.assertEqual(dets[0].team, "enemy")
        self.assertEqual(warned, set())

    def test_costless_spell_skipped_and_warned_once(self):
        db = _db()
        self.assertIsNone(db.elixir("mirror"), "mirror must have no flat cost for this to be the right fixture")
        warned: set = set()
        dets1 = dets_of_costed([_spell_unit("mirror")], db, warned)
        self.assertEqual(dets1, [])
        self.assertEqual(warned, {"mirror"})
        # a second costless mirror det: still skipped, warned set unchanged (warn-ONCE, not per-occurrence)
        dets2 = dets_of_costed([_spell_unit("mirror")], db, warned)
        self.assertEqual(dets2, [])
        self.assertEqual(warned, {"mirror"})

    def test_mixed_batch_keeps_costed_drops_costless(self):
        db = _db()
        warned: set = set()
        dets = dets_of_costed([_spell_unit("fireball"), _spell_unit("mirror")], db, warned)
        self.assertEqual([d.base for d in dets], ["fireball"])
        self.assertEqual(warned, {"mirror"})


# ------------------------------------------------------------------------------------------------------
# (vii) Attempt 3: TrackedEstimator.charged_by_base -- attribution via the wrapped db.elixir() call, the
# ONE place update() ever increments _opp_spent from a detection (opponent_elixir.py:109).
# ------------------------------------------------------------------------------------------------------
class _FakeDB:
    """A synthetic, fixed-cost db -- decoupled from cards.yaml so this test is about the WRAPPING mechanism,
    not about any particular card's real cost."""

    def __init__(self, cost: float = 4.0):
        self._cost = cost

    def elixir(self, name):
        return self._cost


class TestChargedByBase(unittest.TestCase):
    def test_single_enemy_det_charges_exactly_that_base(self):
        est = TrackedEstimator(_FakeDB(4.0))
        est.reset(my_elixir=5.0, now=0.0)
        det = _Det(base="knight", cx=0.5, gy=0.3, team="enemy")
        est.update(5.0, [det], 1.0)
        self.assertEqual(est.charged_by_base, {"knight": [1, 4.0]})
        # and the estimate itself still moved by exactly the cost (TrackedEstimator changes NOTHING about
        # the real update() arithmetic -- it only observes it)
        self.assertAlmostEqual(est._est, 5.0 - 4.0)

    def test_two_different_bases_in_one_update_call_attribute_separately(self):
        """The scenario a before/after ``_opp_spent`` diff CANNOT attribute (module docstring): two
        different new-track bases charged inside the same update() call."""
        est = TrackedEstimator(_FakeDB(3.0))
        est.reset(my_elixir=10.0, now=0.0)
        d1 = _Det(base="knight", cx=0.2, gy=0.3, team="enemy")
        d2 = _Det(base="archers", cx=0.8, gy=0.3, team="enemy")   # far apart -> two distinct tracks, not clustered
        est.update(10.0, [d1, d2], 1.0)
        self.assertEqual(est.charged_by_base, {"knight": [1, 3.0], "archers": [1, 3.0]})
        self.assertAlmostEqual(est._est, 10.0 - 3.0 - 3.0)

    def test_repeat_sighting_of_the_same_track_does_not_recharge(self):
        """A det that MATCHES an existing track (same base, within match_radius) refreshes it instead of
        charging again -- charged_by_base must reflect that, not one entry per update() call."""
        est = TrackedEstimator(_FakeDB(2.0))
        est.reset(my_elixir=10.0, now=0.0)
        det = _Det(base="skeletons", cx=0.5, gy=0.3, team="enemy")
        est.update(10.0, [det], 1.0)
        det_same_spot = _Det(base="skeletons", cx=0.505, gy=0.302, team="enemy")   # within match_radius=0.07
        est.update(10.0, [det_same_spot], 2.0)
        self.assertEqual(est.charged_by_base, {"skeletons": [1, 2.0]})   # still just ONE charge


# ------------------------------------------------------------------------------------------------------
# (viii) Attempt 3: ghost_scripted_by_base -- the SCRIPT, from a hand-built ghost_commands list (reference
# only as of Attempt 4 FIX 1 -- see TestGhostDeliveredByBase for the ledger over-charge actually uses).
# ------------------------------------------------------------------------------------------------------
class TestGhostScriptedByBase(unittest.TestCase):
    def test_built_correctly_from_ghost_commands(self):
        db = _db()
        self.assertEqual(db.elixir("knight"), 3)
        self.assertEqual(db.elixir("musketeer"), 4)
        self.assertEqual(db.elixir("goblin_barrel"), 3)
        # mirror is the confirmed costless fixture (attempt 1's dets-conversion scan: the ONLY held-out
        # ghost-deck base with no CardDB elixir row -- dynamic pricing, no flat `elixir` key). NOTE:
        # giant_snowball looked costless from a naive cards.yaml grep (its curated block has no `elixir`
        # key) but actually resolves to 2 through CardDB's stats-layer merge (cards.py:144-176,
        # cards_stats.json) -- caught by actually running this test rather than assuming from one file,
        # so it is NOT used as the costless fixture here.
        self.assertIsNone(db.elixir("mirror"), "mirror must have no flat cost for this fixture")
        entry = {"ghost_commands": [
            {"card": "knight", "ability": 0},
            {"card": "knight", "ability": 0},
            {"card": "musketeer", "ability": 0},
            {"card": None, "ability": 1},                 # champion ability: excluded, no card
            {"card": "goblin-barrel", "ability": 0},       # hyphenated slug -> goblin_barrel, cost 3
            {"card": "mirror", "ability": 0},              # costless in CardDB -> skipped + warned once
        ]}
        warned: set = set()
        out = ghost_scripted_by_base(entry, db, warned)
        self.assertEqual(out["knight"], [2, 6.0])
        self.assertEqual(out["musketeer"], [1, 4.0])
        self.assertEqual(out["goblin_barrel"], [1, 3.0])
        self.assertNotIn("mirror", out)
        self.assertEqual(warned, {"mirror"})
        self.assertEqual(set(out), {"knight", "musketeer", "goblin_barrel"})

    def test_empty_ghost_commands(self):
        self.assertEqual(ghost_scripted_by_base({"ghost_commands": []}, _db(), set()), {})
        self.assertEqual(ghost_scripted_by_base({}, _db(), set()), {})


# ------------------------------------------------------------------------------------------------------
# (ix) Attempt 4 FIX 1: ghost_delivered_by_base -- the TRUTH ledger over-charge actually uses, built from a
# hand-built env.ghost_cards_delivered-shaped stub, and its divergence from the SCRIPT when a play never
# lands. Ticket's exact scenario: scripted={a:2,b:1}, delivered={a:1} -> ledger {a:[1,cost_a]} ONLY.
# ------------------------------------------------------------------------------------------------------
class TestGhostDeliveredByBase(unittest.TestCase):
    def test_delivered_ledger_differs_from_scripted_when_a_play_is_never_delivered(self):
        db = _FakeDB(3.0)          # every base costs 3.0 -- isolates the ledger mechanics from real card costs
        scripted_entry = {"ghost_commands": [
            {"card": "a", "ability": 0}, {"card": "a", "ability": 0}, {"card": "b", "ability": 0},
        ]}
        scripted = ghost_scripted_by_base(scripted_entry, db)
        self.assertEqual(scripted, {"a": [2, 6.0], "b": [1, 3.0]})       # the script: a x2, b x1

        delivered_stub = {"a": 1}          # env.ghost_cards_delivered-shaped: only ONE 'a' ever landed, no 'b'
        delivered = ghost_delivered_by_base(delivered_stub, db)
        self.assertEqual(delivered, {"a": [1, 3.0]})                     # ONLY this -- exactly the ticket's spec
        self.assertNotIn("b", delivered)
        self.assertNotEqual(delivered, scripted)                         # the two ledgers must actually diverge

    def test_slug_normalization_and_costless_skip_match_scripted(self):
        db = _db()
        delivered_stub = {"goblin-barrel": 2, "mirror": 1}   # hyphenated slug + a costless one (Counter-shaped)
        warned: set = set()
        out = ghost_delivered_by_base(delivered_stub, db, warned)
        self.assertEqual(out, {"goblin_barrel": [2, 6.0]})
        self.assertEqual(warned, {"mirror"})

    def test_empty_and_none_counts_are_skipped(self):
        self.assertEqual(ghost_delivered_by_base({}, _db()), {})
        self.assertEqual(ghost_delivered_by_base({"knight": 0}, _db()), {})   # a zero count is a no-op


# ------------------------------------------------------------------------------------------------------
# (x) Attempt 4 FIX 2: opp_play_flags -- flags only the first sample tick at/after a DELIVERED ghost event.
# Ticket's exact scenario: a 1-cost card delivered at tick 40 under 5-tick stepping -> ONLY the tick-40 row.
# ------------------------------------------------------------------------------------------------------
class TestOppPlayFlags(unittest.TestCase):
    def test_delivery_at_tick_40_flags_only_that_sample(self):
        sample_ticks = [30, 35, 40, 45, 50]                     # 5-tick stepping, matches run_match_audit
        ghost_events = [(40, 1, "accepted")]                    # a 1-cost card delivered exactly at tick 40
        got = opp_play_flags(sample_ticks, ghost_events)
        self.assertEqual(got, [False, False, True, False, False])

    def test_refused_delivery_never_flags(self):
        sample_ticks = [30, 35, 40, 45, 50]
        ghost_events = [(40, 0, "not_enough_elixir")]           # REFUSED -- accepted==0, must not flag anything
        self.assertEqual(opp_play_flags(sample_ticks, ghost_events), [False] * 5)

    def test_delivery_between_samples_flags_the_next_one(self):
        sample_ticks = [30, 35, 40, 45, 50]
        ghost_events = [(37, 1, "accepted")]                    # falls strictly between 35 and 40
        got = opp_play_flags(sample_ticks, ghost_events)
        self.assertEqual(got, [False, False, True, False, False])

    def test_two_deliveries_in_one_gap_flag_the_same_shared_sample(self):
        sample_ticks = [30, 35, 40, 45, 50]
        ghost_events = [(36, 1, "accepted"), (38, 1, "accepted")]   # both between 35 and 40
        got = opp_play_flags(sample_ticks, ghost_events)
        self.assertEqual(got, [False, False, True, False, False])  # ONE flagged row, not an error / two rows

    def test_delivery_before_first_sample_flags_first_sample(self):
        sample_ticks = [30, 35, 40]
        ghost_events = [(10, 1, "accepted")]
        self.assertEqual(opp_play_flags(sample_ticks, ghost_events), [True, False, False])

    def test_no_events(self):
        self.assertEqual(opp_play_flags([30, 35, 40], []), [False, False, False])


# ------------------------------------------------------------------------------------------------------
# (xi) Attempt 4 FIX 4: dets_of_whitelisted drops a non-whitelisted enemy base, keeps a whitelisted one;
# load_detector_cards loads the real, non-empty live whitelist the same way play.py:376 does.
# ------------------------------------------------------------------------------------------------------
class TestDetectorCardsWhitelist(unittest.TestCase):
    def test_load_detector_cards_measured_membership(self):
        deck = load_deck("icebow")
        wl = load_detector_cards(deck.config)
        self.assertIsInstance(wl, frozenset)
        self.assertGreater(len(wl), 0)
        # MEASURED (this test, independently of the verifier): knight is in the whitelist, golem is not --
        # both directly checked against the live icebow/config/config.yaml, not assumed.
        self.assertIn("knight", wl)
        self.assertNotIn("golem", wl)
        # MEASURED correction of the Attempt-2 docstring's "zero spell entries" claim: the_log IS a spell
        # (vocab.SPELL_CLASSES) and IS in the whitelist.
        self.assertIn("the_log", wl)
        self.assertIn("the_log", vocab.SPELL_CLASSES)

    def test_dets_of_whitelisted_drops_outside_keeps_inside(self):
        wl = frozenset({"knight"})
        units = [_spell_unit("knight", side=1), _spell_unit("golem", side=1)]
        dets = dets_of_whitelisted(units, wl)
        self.assertEqual([d.base for d in dets], ["knight"])

    def test_dets_of_whitelisted_never_filters_non_enemy_team(self):
        """A mine/unknown det is never dropped by the whitelist filter -- the estimator's own team=='enemy'
        check in update() handles those regardless, so this filter only ever needs to act on enemy dets."""
        wl = frozenset({"knight"})
        units = [_spell_unit("golem", side=0), _spell_unit("golem", side=-1)]   # golem, but mine / unknown
        dets = dets_of_whitelisted(units, wl)
        self.assertEqual(len(dets), 2)


# ------------------------------------------------------------------------------------------------------
# (xii) Attempt 4 FIX 5: overcharge_table / merge_base_ledgers on hand-built ledgers, including a base
# present ONLY in charges (never played by the ghost at all -- the spawned-units signature).
# ------------------------------------------------------------------------------------------------------
class TestOvercharge(unittest.TestCase):
    def test_merge_base_ledgers_sums_across_matches(self):
        m1 = {"knight": [1, 3.0], "golem": [1, 8.0]}
        m2 = {"knight": [2, 6.0]}
        out = merge_base_ledgers([m1, m2])
        self.assertEqual(out, {"knight": [3, 9.0], "golem": [1, 8.0]})

    def test_merge_base_ledgers_empty_list(self):
        self.assertEqual(merge_base_ledgers([]), {})

    def test_overcharge_table_base_never_played_shows_full_charge_as_overcharge(self):
        # "golemite" is charged twice (a spawned body) but NEVER appears in the ghost's played-card ledger --
        # exactly the spawned-units signature the coordinator's hypothesis predicts.
        charged = {"knight": [1, 3.0], "golemite": [2, 2.0]}
        ghost = {"knight": [1, 3.0]}                     # ghost only ever played knight
        oc = overcharge_table(charged, ghost, top=25)
        by_base = {r["base"]: r for r in oc["top"]}
        self.assertEqual(by_base["knight"]["over_charge"], 0.0)
        self.assertEqual(by_base["golemite"]["ghost_plays"], 0)
        self.assertEqual(by_base["golemite"]["over_charge"], 2.0)        # charged_elixir - ghost_elixir(0)
        self.assertEqual(oc["total_charged_elixir"], 5.0)
        self.assertEqual(oc["total_ghost_elixir"], 3.0)
        self.assertEqual(oc["total_over_charge"], 2.0)
        # ALL of the over-charge here comes from a base the ghost never played:
        self.assertAlmostEqual(oc["over_charge_share_from_bases_never_played"], 2.0 / 5.0)

    def test_overcharge_table_sorted_by_abs_over_charge_desc_and_top_n(self):
        charged = {"a": [1, 1.0], "b": [1, 10.0], "c": [1, 5.0]}
        ghost = {"a": [1, 1.0], "b": [1, 0.0], "c": [1, 5.0]}   # over_charge: a=0, b=10, c=0
        oc = overcharge_table(charged, ghost, top=2)
        self.assertEqual(len(oc["top"]), 2)
        self.assertEqual(oc["top"][0]["base"], "b")             # the only nonzero |over_charge| leads
        self.assertEqual(oc["n_bases"], 3)

    def test_overcharge_table_no_bases(self):
        oc = overcharge_table({}, {})
        self.assertEqual(oc["top"], [])
        self.assertEqual(oc["total_charged_elixir"], 0.0)
        self.assertIsNone(oc["over_charge_share_from_bases_never_played"])   # 0/0 -- None, not a ZeroDivisionError


# ------------------------------------------------------------------------------------------------------
# (xiii) TICKET O10: charge-event trace -- TracedEstimator / run_traced_update against synthetic dets, no
# engine. The three ticket-mandated scenarios (stationary/first_seen, jump/rebill_out_of_radius,
# expire-then-reappear/rebill_after_expiry), plus TracedEstimator parity with plain TrackedEstimator.
# ------------------------------------------------------------------------------------------------------
class TestChargeTraceStationaryFirstSeen(unittest.TestCase):
    def test_three_samples_0_25s_apart_one_charge_zero_expiries(self):
        est = TracedEstimator(_FakeDB(4.0))
        est.reset(my_elixir=10.0, now=0.0)
        t_to_tick: dict = {}
        all_charges: list = []
        all_tracks: list = []
        prev_dets, prev_t = None, None
        for i, t in enumerate((0.0, 0.25, 0.5)):
            det = _Det(base="tombstone", cx=0.4, gy=0.6, team="enemy")
            ch, tr = run_traced_update(est, 10.0, [det], t, tick=i * 5, tag="m1", k=0,
                                       prev_dets=prev_dets, prev_t=prev_t, t_to_tick=t_to_tick)
            all_charges.extend(ch)
            all_tracks.extend(tr)
            prev_dets, prev_t = [det], t
        self.assertEqual(len(all_charges), 1, all_charges)
        self.assertEqual(all_charges[0]["reason"], "first_seen")
        self.assertEqual(all_charges[0]["base"], "tombstone")
        self.assertEqual(all_charges[0]["cost"], 4.0)
        self.assertEqual(all_charges[0]["prev_n_same_base"], 0)
        self.assertFalse(all_charges[0]["same_base_track_existed_before"])
        self.assertIsNone(all_charges[0]["dt_since_prev_sample_s"])   # first sample of the match
        self.assertEqual(all_tracks, [])   # zero track_events -- the track never aged past forget_s


class TestChargeTraceJumpRebillOutOfRadius(unittest.TestCase):
    def test_jump_0_2_in_x_produces_second_charge_rebill_out_of_radius(self):
        est = TracedEstimator(_FakeDB(3.0))
        est.reset(my_elixir=10.0, now=0.0)
        t_to_tick: dict = {}
        det1 = _Det(base="miner", cx=0.5, gy=0.5, team="enemy")
        ch1, tr1 = run_traced_update(est, 10.0, [det1], 0.0, tick=0, tag="m1", k=0,
                                     prev_dets=None, prev_t=None, t_to_tick=t_to_tick)
        self.assertEqual(len(ch1), 1)
        self.assertEqual(ch1[0]["reason"], "first_seen")

        det2 = _Det(base="miner", cx=0.7, gy=0.5, team="enemy")     # jumped 0.2 in x -- > match_radius=0.07
        ch2, tr2 = run_traced_update(est, 10.0, [det2], 0.25, tick=5, tag="m1", k=0,
                                     prev_dets=[det1], prev_t=0.0, t_to_tick=t_to_tick)
        self.assertEqual(len(ch2), 1, ch2)
        row = ch2[0]
        self.assertEqual(row["reason"], "rebill_out_of_radius")
        self.assertEqual(row["prev_n_same_base"], 1)
        self.assertAlmostEqual(row["prev_same_base_min_dist"], 0.2, places=4)
        self.assertFalse(row["prev_same_base_within_match_radius"])
        self.assertTrue(row["same_base_track_existed_before"])
        self.assertAlmostEqual(row["dt_since_prev_sample_s"], 0.25, places=4)
        self.assertEqual(tr2, [])   # the old track was never expired (age 0.25 <= forget_s=6.0), just missed


class TestChargeTraceExpiryRebillAfterExpiry(unittest.TestCase):
    def test_absent_7s_then_present_produces_expiry_and_rebill_after_expiry(self):
        est = TracedEstimator(_FakeDB(2.0))
        est.reset(my_elixir=10.0, now=0.0)
        t_to_tick: dict = {}
        det = _Det(base="witch", cx=0.3, gy=0.3, team="enemy")
        ch1, tr1 = run_traced_update(est, 10.0, [det], 0.0, tick=0, tag="m1", k=0,
                                     prev_dets=None, prev_t=None, t_to_tick=t_to_tick)
        self.assertEqual(ch1[0]["reason"], "first_seen")
        self.assertEqual(tr1, [])

        # next OBSERVED sample is 7.0s later (unit absent from every intervening sample this test skips
        # feeding, i.e. never detected in between) -- forget_s=6.0 means the track is stale by then, and the
        # real _seen_tracks (opponent_elixir.py:42-43) removes it INSIDE this same update() call, before the
        # reappearing det is matched against it.
        det_again = _Det(base="witch", cx=0.3, gy=0.3, team="enemy")   # same spot -- stationary building
        ch2, tr2 = run_traced_update(est, 10.0, [det_again], 7.0, tick=140, tag="m1", k=0,
                                     prev_dets=[det], prev_t=0.0, t_to_tick=t_to_tick)
        self.assertEqual(len(tr2), 1, tr2)
        self.assertEqual(tr2[0]["base"], "witch")
        self.assertAlmostEqual(tr2[0]["age_s"], 7.0, places=4)
        self.assertEqual(tr2[0]["last_seen_tick"], 0)      # resolved via t_to_tick, set at tick=0's call

        self.assertEqual(len(ch2), 1, ch2)
        row = ch2[0]
        self.assertEqual(row["reason"], "rebill_after_expiry")
        self.assertTrue(row["same_base_track_existed_before"])   # true BEFORE this update's own expiry
        self.assertEqual(row["prev_n_same_base"], 1)              # stationary -- also seen in prev sample
        self.assertAlmostEqual(row["prev_same_base_min_dist"], 0.0, places=6)


class TestChargeTraceOffProducesNoTrace(unittest.TestCase):
    def test_charge_trace_false_path_uses_plain_update_no_trace_helpers_needed(self):
        """Sanity check for the --charge-trace OFF path: run_match_audit's est_a stays a plain
        TrackedEstimator (no last_charges/last_expired) when charge_trace=False -- verified here at the
        estimator-selection unit, since run_match_audit itself needs a live engine to exercise end to end."""
        est_off = TrackedEstimator(_FakeDB(3.0))
        self.assertFalse(hasattr(est_off, "last_charges"))
        self.assertFalse(hasattr(est_off, "last_expired"))
        est_on = TracedEstimator(_FakeDB(3.0))
        self.assertTrue(hasattr(est_on, "last_charges"))
        self.assertTrue(hasattr(est_on, "last_expired"))


class TestTracedEstimatorParityWithTrackedEstimator(unittest.TestCase):
    def test_same_sequence_of_updates_yields_identical_est_and_charged_by_base(self):
        """CONSTRAINT: 'no change to opponent_elixir.py semantics; the wrapper must call the real
        update()/_seen_tracks so condition A's numbers are unchanged.' Feeds the identical sequence of dets
        to a TrackedEstimator and a TracedEstimator and asserts identical numeric outcomes."""
        seq = [
            (10.0, [_Det("knight", 0.5, 0.3, "enemy")], 0.0),
            (10.0, [_Det("knight", 0.505, 0.302, "enemy")], 0.25),          # refresh, within match_radius
            (10.0, [_Det("knight", 0.9, 0.3, "enemy"), _Det("archers", 0.1, 0.3, "enemy")], 0.5),
            (10.0, [], 8.0),                                                # gap -- lets knight's track expire
            (10.0, [_Det("knight", 0.9, 0.3, "enemy")], 8.25),
        ]
        base = TrackedEstimator(_FakeDB(2.0))
        traced = TracedEstimator(_FakeDB(2.0))
        base.reset(my_elixir=10.0, now=0.0)
        traced.reset(my_elixir=10.0, now=0.0)
        for my_elx, dets, now in seq:
            base.update(my_elx, dets, now)
            traced.update(my_elx, dets, now)
        self.assertAlmostEqual(base._est, traced._est)
        self.assertEqual(base.charged_by_base, traced.charged_by_base)
        self.assertEqual(base._opp_spent, traced._opp_spent)


if __name__ == "__main__":
    unittest.main()
