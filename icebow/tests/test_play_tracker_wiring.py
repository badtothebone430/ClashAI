"""L67y T1 (HANDOFF 5cs.99 AD): live play's TeamTracker gets the filters train-rl's env.py always had.

play.py built its TeamTracker without `is_spell`, so enemy SPELL detections were served to the aim assists -- among
them a false enemy "earthquake" the detector reads on our own king tower (48 of 270 captured states), which the
Tornado king-activation assist aimed at (41 of 106 run12 Tornados redirected). Pinned two ways: the wiring in
play.py's source, and the behaviour of a tracker built with play.py's exact filter.
"""
from __future__ import annotations

import ast
import itertools
import os
import re
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clashrl import card_threat, play                                     # noqa: E402
from clashrl.cards import CardDB                                          # noqa: E402
from clashrl.config import Config                                         # noqa: E402
from clashrl.opponent_elixir import OpponentElixirEstimator, OpponentElixirEstimatorV2  # noqa: E402
from clashrl.perception import PerceptionLoop                            # noqa: E402
from clashrl.replay_mine import Detection, TeamTracker, own_card_bases    # noqa: E402

PLAY = os.path.join(os.path.dirname(__file__), "..", "src", "clashrl", "play.py")


def _construction() -> str:
    with open(PLAY, encoding="utf-8") as fh:
        src = fh.read()
    i = src.index("_team_tracker = TeamTracker(")
    depth, j = 0, src.index("(", i)
    for k in range(j, len(src)):
        depth += {"(": 1, ")": -1}.get(src[k], 0)
        if depth == 0:
            return src[i:k + 1]
    raise AssertionError("unbalanced TeamTracker( call in play.py")


def _det(base, x, y, team="unknown"):
    return Detection(base, x, y, 0.05, 0.05, 0.9, team, None, None, None)


class PlayTrackerWiring(unittest.TestCase):
    def test_play_passes_the_three_filters_env_py_passes(self):
        call = _construction()
        for kw in ("is_spell=", "min_hits=", "phantom_stale_s="):
            self.assertIn(kw, call, f"play.py's TeamTracker is missing {kw}")
        self.assertTrue(re.search(r'is_spell=lambda b, _db=_db: _db\.kind\(card_threat\.base_key\(str\(b\)\)\) == "spell"', call),
                        "play.py's is_spell is not the base-folded CardDB spell check this test exercises")


class PlaysSpellFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = CardDB(Config.load())

    def _tracker(self):
        db = self.db
        return TeamTracker(own_cards=own_card_bases(db), min_hits=2, phantom_stale_s=6.0,
                           is_spell=lambda b, _db=db: _db.kind(card_threat.base_key(str(b))) == "spell")

    def _served(self, base, x, y):
        tr = self._tracker()
        tr.tag([_det(base, x, y)], 0.0)
        tr.tag([_det(base, x, y + 0.005)], 0.15)                 # corroborated: min_hits is not what removes it
        return [t[4] for t in tr.enemy_tracks(0.2, True)]

    def test_the_false_enemy_earthquake_on_our_king_is_not_served(self):
        self.assertEqual(self._served("earthquake", 0.495, 0.72), [])

    def test_an_aoe_ring_class_name_is_folded_and_not_served(self):
        self.assertEqual(self._served("earthquake_aoe", 0.495, 0.72), [])

    def test_a_real_enemy_troop_is_still_served(self):
        # a troop in NEITHER deck (hogeq holds hog_rider), so this file stays byte-identical across the two trees
        self.assertEqual(self._served("giant", 0.50, 0.25), ["giant"])

    def test_a_spawn_spell_is_still_served(self):
        self.assertEqual(self._served("goblin_barrel", 0.30, 0.55), ["goblin_barrel"])

    def test_without_the_filter_an_enemy_spell_on_our_king_WAS_served(self):
        # fireball: a spell in NEITHER deck (hogeq owns earthquake, so its deck veto would not call one "enemy")
        tr = TeamTracker(own_cards=own_card_bases(self.db), min_hits=2, phantom_stale_s=6.0)     # play.py before T1
        tr.tag([_det("fireball", 0.495, 0.72)], 0.0)
        tr.tag([_det("fireball", 0.495, 0.725)], 0.15)
        self.assertEqual([t[4] for t in tr.enemy_tracks(0.2, True)], ["fireball"])
        self.assertEqual(self._served("fireball", 0.495, 0.72), [])                               # and with T1 it is not


def _src() -> str:
    with open(PLAY, encoding="utf-8") as fh:
        return fh.read()


def _bill_state() -> dict:
    return {"uid_counter": itertools.count(1), "billed": set()}


def _est_tracker(min_hits=2, forget_s=6.0):
    """A TeamTracker with play.py's own is_spell filter and min_hits -- same convention as
    PlaysSpellFilter._tracker() above, extended with a settable forget_s for the expiry test."""
    db = CardDB(Config.load())
    return TeamTracker(own_cards=own_card_bases(db), min_hits=min_hits, forget_s=forget_s,
                       phantom_stale_s=6.0,
                       is_spell=lambda b, _db=db: _db.kind(card_threat.base_key(str(b))) == "spell")


def _rhs_of(name: str) -> str:
    """The RHS expression text of play()'s (single-line) top-level `name = ...` assignment -- balanced on
    brackets so a call/ternary is captured whole, stopping at the first depth-0 newline. Lets a test eval()
    the REAL expression from play.py's own source against a controlled namespace, instead of grepping for
    it as a fixed string -- behavioural in the sense that it exercises the actual code, still source-located
    because play() itself cannot be invoked in a unit test (WindowCapture/Controller/torch device need a
    live window)."""
    src = _src()
    marker = f"{name} = "
    i = src.index(marker)
    j = i + len(marker)
    depth, k = 0, j
    while k < len(src):
        c = src[k]
        depth += {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}.get(c, 0)
        if depth <= 0 and c == "\n":
            break
        k += 1
    return src[j:k]


class _RecordMyPlayGuardVisitor(ast.NodeVisitor):
    """Walks play.py's AST tracking the stack of enclosing `if` TESTS, and records that stack at the (sole
    expected) `_opp_elx.record_my_play(...)` call site -- an AST walk finds the call's TRUE enclosing
    control flow regardless of comment text/line spacing around it, unlike a fixed-window text grep."""

    def __init__(self):
        self.if_stack: list = []
        self.enclosing_tests: "list | None" = None
        self.n_calls = 0

    def visit_If(self, node):
        self.if_stack.append(node.test)
        self.generic_visit(node)
        self.if_stack.pop()

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "record_my_play"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "_opp_elx"):
            self.n_calls += 1
            self.enclosing_tests = list(self.if_stack)
        self.generic_visit(node)


class OppElixirV2Wiring(unittest.TestCase):
    """O19: the two independent switches -- which estimator COMPUTES the number (play.opp_elixir_v2) vs
    whether it REACHES S1 (play.student_opp_elixir, pre-existing) -- and the OFF path staying byte-identical
    to before this ticket. play() itself cannot be called in a unit test (WindowCapture/Controller/torch
    device need a live window), so every test here is source-located one way or another; where the ACTUAL
    expression can be eval()'d against a controlled namespace (construction, the S1 gate, the flag default)
    that is done instead of a plain string match. Two facts are pure CONTROL-FLOW/ORDERING claims with no
    expression to evaluate in isolation (the OFF branch's update() call has no free-standing local variable
    to eval it against outside play() itself; a source-order comparison between two reset lines is only
    meaningful in the real match-reset control flow) -- those two stay literal text pins, same convention as
    the pre-existing PlayTrackerWiring.test_play_passes_the_three_filters_env_py_passes above, and are
    labelled TEXT PIN below."""

    def test_the_flag_reads_like_student_opp_elixir_and_really_defaults_to_False(self):
        rhs = _rhs_of("_opp_elixir_v2")
        cfg = Config.load()                      # config.yaml does not set play.opp_elixir_v2
        self.assertIs(eval(rhs, {}, {"cfg": cfg, "bool": bool}), False)

    def test_off_path_construction_actually_builds_v1_not_v2(self):
        ns = {"OpponentElixirEstimator": OpponentElixirEstimator,
              "OpponentElixirEstimatorV2": OpponentElixirEstimatorV2,
              "_db": CardDB(Config.load()), "_opp_elixir_v2": False}
        obj = eval(_rhs_of("_opp_elx"), {}, ns)
        self.assertIsInstance(obj, OpponentElixirEstimator)
        self.assertNotIsInstance(obj, OpponentElixirEstimatorV2)

    def test_on_path_construction_actually_builds_v2(self):
        ns = {"OpponentElixirEstimator": OpponentElixirEstimator,
              "OpponentElixirEstimatorV2": OpponentElixirEstimatorV2,
              "_db": CardDB(Config.load()), "_opp_elixir_v2": True}
        obj = eval(_rhs_of("_opp_elx"), {}, ns)
        self.assertIsInstance(obj, OpponentElixirEstimatorV2)

    def test_off_path_update_call_is_unchanged(self):
        # TEXT PIN: no local variable exists to eval this call's argument list against outside play()
        # itself; see the class docstring.
        self.assertIn('_est = _opp_elx.update(float(my_elixir), dets, now)', _src(),
                      "the OFF branch must call update() with exactly (my_elixir, dets, now), as before")

    def test_on_path_bills_via_the_shared_tracker_and_reapplies_the_whitelist(self):
        src = _src()
        self.assertIn('_opp_elixir_v2_bill(_team_tracker, _opp_bill)', src,
                      "must reuse play.py's own _team_tracker, not build a second one")
        self.assertNotIn("TeamTracker(", src[src.index('if _opp_elixir_v2 or _opp_elixir_v2_shadow:'):
                                              src.index('mem[5] = _est')],
                         "a second TeamTracker must not be constructed on the billing path")
        i = src.index('_opp_elixir_v2_bill(_team_tracker, _opp_bill)')
        j = src.index('_est = _opp_elx.update(float(my_elixir), _billed, now)')
        self.assertIn('d.base in detector_cards', src[i:j],
                      "the billed dets must be re-filtered through the SAME detector_cards whitelist")

    def test_match_reset_clears_the_billed_set_alongside_the_estimator_reset(self):
        # TEXT PIN (source ordering): meaningful only in the real per-match reset control flow; see the
        # class docstring.
        src = _src()
        i = src.index('_opp_elx.reset(my_elixir=float(vision.read_elixir(frame)), now=time.time())')
        j = src.index('_opp_bill["billed"].clear()')
        self.assertLess(i, j)
        self.assertLess(j - i, 320, "the billed-set reset should sit right next to the estimator reset")

    def test_student_opp_elixir_is_untouched_and_independent_of_the_new_flag(self):
        """(vi) both flags OFF -> S1 still sees None. BEHAVIOURAL: eval() the real `_oe = ...` expression in
        a namespace that DELIBERATELY OMITS `_opp_elixir_v2` -- if the code ever came to depend on that name
        (e.g. `... if (_student_opp_elixir or _opp_elixir_v2) else None`), this raises NameError instead of
        silently passing a substring check."""
        rhs = _rhs_of("_oe")

        class _Stub:
            _est = 3.5

        ns = {"_opp_elx": _Stub(), "_student_opp_elixir": False}
        self.assertIsNone(eval(rhs, {}, ns))
        ns["_student_opp_elixir"] = True
        self.assertEqual(eval(rhs, {}, ns), 3.5)

    def test_record_my_play_is_unconditional_in_both_modes(self):
        v = _RecordMyPlayGuardVisitor()
        v.visit(ast.parse(_src()))
        self.assertEqual(v.n_calls, 1, "expected exactly one _opp_elx.record_my_play(...) call in play.py")
        guarded_on_v2 = any("_opp_elixir_v2" in ast.dump(t) for t in v.enclosing_tests)
        self.assertFalse(guarded_on_v2,
                         "record_my_play must not be gated behind opp_elixir_v2 -- it must fire in both modes")


class OppElixirV2Billing(unittest.TestCase):
    """Behavioural half of O19: the actual (importable, module-level) billing helper against a real
    TeamTracker built with play.py's own filters -- not a re-implementation of play.py's logic."""

    def test_a_one_frame_phantom_is_never_billed(self):
        tr = _est_tracker(min_hits=2)
        bs = _bill_state()
        tr.tag([_det("giant", 0.5, 0.25)], 0.0)     # 1 sighting -- below min_hits
        self.assertEqual(play._opp_elixir_v2_bill(tr, bs), [])

    def test_a_two_frame_unit_is_billed_exactly_once_at_its_second_sighting(self):
        tr = _est_tracker(min_hits=2)
        bs = _bill_state()
        tr.tag([_det("giant", 0.5, 0.25)], 0.0)
        self.assertEqual(play._opp_elixir_v2_bill(tr, bs), [], "1st sighting: hits=1, not yet confirmed")
        tr.tag([_det("giant", 0.5, 0.255)], 0.15)   # 2nd sighting -- now confirmed
        out = play._opp_elixir_v2_bill(tr, bs)
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], play._BilledDet)
        self.assertEqual((out[0].base, out[0].team), ("giant", "enemy"))
        # a later sighting of the SAME (still-live) track must not bill it a second time
        tr.tag([_det("giant", 0.5, 0.26)], 0.30)
        self.assertEqual(play._opp_elixir_v2_bill(tr, bs), [])

    def test_an_expired_then_recreated_track_is_billed_again(self):
        tr = _est_tracker(min_hits=2, forget_s=0.5)
        bs = _bill_state()
        tr.tag([_det("giant", 0.5, 0.25)], 0.0)
        tr.tag([_det("giant", 0.5, 0.255)], 0.1)
        first = play._opp_elixir_v2_bill(tr, bs)
        self.assertEqual(len(first), 1)
        # gap > forget_s (0.5s): the old track is dropped by tag() before this next call
        tr.tag([_det("giant", 0.5, 0.25)], 5.0)
        tr.tag([_det("giant", 0.5, 0.255)], 5.1)
        second = play._opp_elixir_v2_bill(tr, bs)
        self.assertEqual(len(second), 1, "an expired-then-recreated track must bill again")

    def test_the_whitelist_still_filters_billed_dets(self):
        """(iv) the detector_cards whitelist still applies in ON mode -- reproduces play.py's own
        `[d for d in _opp_elixir_v2_bill(...) if d.base in detector_cards]` line against a confirmed
        off-whitelist track."""
        tr = _est_tracker(min_hits=2)
        bs = _bill_state()
        tr.tag([_det("giant", 0.5, 0.25)], 0.0)
        tr.tag([_det("giant", 0.5, 0.255)], 0.15)
        billed = play._opp_elixir_v2_bill(tr, bs)
        self.assertEqual(len(billed), 1)
        detector_cards = {"hog_rider", "musketeer"}          # a whitelist that excludes "giant"
        self.assertEqual([d for d in billed if d.base in detector_cards], [])

    def test_v1_and_v2_estimators_charge_the_billed_dets_own_known_cost(self):
        """(FIX 3) Both estimator classes clip their output to [0, 1] UNCONDITIONALLY -- a bare range check
        would pass even for a _BilledDet missing .base/.cx entirely. Assert the estimator actually CONSUMED
        the billed det's fields instead: one confirmed 'giant' det must charge exactly giant's own known
        elixir cost (`giant` has no SPAWNS/BODIES entry and no sibling track, so neither class's extra rules
        change this from V1's plain accounting -- both must agree)."""
        db = CardDB(Config.load())
        cost = db.elixir("giant")
        self.assertTrue(cost and cost > 0, "giant must have a known positive elixir cost for this test")
        tr = _est_tracker(min_hits=2)
        bs = _bill_state()
        tr.tag([_det("giant", 0.5, 0.25)], 0.0)
        tr.tag([_det("giant", 0.5, 0.255)], 0.15)
        billed = play._opp_elixir_v2_bill(tr, bs)
        self.assertEqual(len(billed), 1)
        for est in (OpponentElixirEstimator(db), OpponentElixirEstimatorV2(db)):
            est.reset(my_elixir=5.0, now=0.0)
            est.update(5.0, billed, 0.15)
            self.assertAlmostEqual(est._opp_spent, float(cost), places=6,
                                   msg=f"{type(est).__name__} did not charge the billed det's own cost")
            self.assertAlmostEqual(est._est, 5.0 - float(cost), places=6)


class _Cfg:
    """Same minimal cfg stub as test_perception_with_base_i9.py (this file's PerceptionLoop is built
    directly, not through play(), so it never calls anything but .get on it)."""

    def get(self, *a, **k):
        return k.get("default")


def _confirmed_tracker(base="giant", hits=2):
    tk = TeamTracker()
    now = time.time()
    tk._tracks = [{"team": "enemy", "t": now, "t0": now - 1.0, "x": 0.5, "y": 0.25,
                  "x0": 0.5, "y0": 0.25, "hits": hits, "base": base, "bm": 0, "be": 0, "rank": 0}]
    return tk


class PerceptionBillConfirmedLockTests(unittest.TestCase):
    """O19 attempt 2 FIX 1 (verifier, MEDIUM): PerceptionLoop.bill_confirmed must run the WHOLE billing
    scan inside its own lock (the same lock its _run() thread holds while mutating the tracker's track
    dicts in place), and must NOT still be holding that lock once it returns (the estimator's own update()
    runs after, unlocked, in play.py)."""

    def test_the_scan_runs_while_the_lock_is_held(self):
        import clashrl.perception as perception_mod
        loop = PerceptionLoop(_Cfg(), None, _confirmed_tracker(), conf=0.5)
        seen_locked = []
        orig = perception_mod._opp_elixir_v2_bill

        def _spy(tracker, bill_state):
            seen_locked.append(loop._lock.locked())
            return orig(tracker, bill_state)

        perception_mod._opp_elixir_v2_bill = _spy
        try:
            loop.bill_confirmed(_bill_state())
        finally:
            perception_mod._opp_elixir_v2_bill = orig
        self.assertEqual(seen_locked, [True], "the billing scan must run while self._lock is held")

    def test_the_lock_is_released_before_bill_confirmed_returns(self):
        loop = PerceptionLoop(_Cfg(), None, _confirmed_tracker(), conf=0.5)
        loop.bill_confirmed(_bill_state())
        self.assertFalse(loop._lock.locked(),
                         "the estimator's update() runs after this returns and must not run while locked")

    def test_bill_confirmed_agrees_with_the_unlocked_helper_on_the_same_tracker(self):
        tk = _confirmed_tracker()
        loop = PerceptionLoop(_Cfg(), None, tk, conf=0.5)
        locked_out = loop.bill_confirmed(_bill_state())
        unlocked_out = play._opp_elixir_v2_bill(tk, _bill_state())
        self.assertEqual(len(locked_out), 1)
        self.assertEqual((locked_out[0].base, locked_out[0].team),
                         (unlocked_out[0].base, unlocked_out[0].team))

    def test_a_one_hit_track_is_not_billed_through_the_passthrough_either(self):
        loop = PerceptionLoop(_Cfg(), None, _confirmed_tracker(hits=1), conf=0.5)
        self.assertEqual(loop.bill_confirmed(_bill_state()), [])

    def test_the_whitelist_is_applied_after_the_locked_scan(self):
        loop = PerceptionLoop(_Cfg(), None, _confirmed_tracker(base="giant"), conf=0.5)
        self.assertEqual(loop.bill_confirmed(_bill_state(), whitelist={"knight"}), [],
                         "an off-whitelist base must still be dropped")
        self.assertEqual(len(loop.bill_confirmed(_bill_state(), whitelist={"giant"})), 1)


class OppElixirV2ShadowWiring(unittest.TestCase):
    """O19 attempt 2 FIX 2: play.opp_elixir_v2_shadow is OBSERVE-ONLY -- it must be able to compute V2
    beside whichever estimator actually drives mem[5]/S1, but must never itself be able to reach either.
    Same play()-cannot-be-called caveat as OppElixirV2Wiring above; the driving/print-format logic that
    also lives inside the same `if _opp_elixir_v2_shadow:` block is left a TEXT PIN for that reason."""

    def test_the_shadow_flag_defaults_off(self):
        rhs = _rhs_of("_opp_elixir_v2_shadow")
        cfg = Config.load()                      # config.yaml does not set play.opp_elixir_v2_shadow
        self.assertIs(eval(rhs, {}, {"cfg": cfg, "bool": bool}), False)

    def test_shadow_construction_is_always_the_class_OTHER_than_opp_elx(self):
        ns = {"OpponentElixirEstimator": OpponentElixirEstimator,
              "OpponentElixirEstimatorV2": OpponentElixirEstimatorV2,
              "_db": CardDB(Config.load())}
        rhs = _rhs_of("_opp_elx_shadow")
        for v2_flag, expect in ((False, OpponentElixirEstimatorV2), (True, OpponentElixirEstimator)):
            ns["_opp_elixir_v2"], ns["_opp_elixir_v2_shadow"] = v2_flag, True
            self.assertIsInstance(eval(rhs, {}, ns), expect)
        ns["_opp_elixir_v2_shadow"] = False
        self.assertIsNone(eval(rhs, {}, ns), "no shadow object unless the shadow flag is on")

    def test_mem5_and_s1_lines_never_mention_the_shadow_estimator(self):
        src = _src()
        mem5_line = src.splitlines()[src[:src.index("mem[5] = _est")].count("\n")]
        self.assertNotIn("shadow", mem5_line.lower())
        oe_rhs = _rhs_of("_oe")
        self.assertNotIn("shadow", oe_rhs.lower())

    def test_shadow_estimator_is_reset_alongside_the_primary_one(self):
        # TEXT PIN (source ordering, same reasoning as OppElixirV2Wiring's own reset-ordering test).
        src = _src()
        self.assertIn("if _opp_elx_shadow is not None:", src)
        i = src.index('_opp_elx.reset(my_elixir=float(vision.read_elixir(frame)), now=time.time())')
        j = src.index("_opp_elx_shadow.reset(")
        self.assertLess(i, j)
        self.assertLess(j - i, 200)

    def test_billing_input_is_shared_by_v2_and_shadow_v2(self):
        """When either flag needs billed dets, both use the SAME `_opp_elixir_v2_bill(_team_tracker,
        _opp_bill)` call -- one scan feeds both estimators, not a second independent tracker walk."""
        src = _src()
        self.assertIn('if _opp_elixir_v2 or _opp_elixir_v2_shadow:', src)


if __name__ == "__main__":
    unittest.main()
