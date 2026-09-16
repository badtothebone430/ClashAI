"""O13: OpponentElixirEstimatorV2 -- hand-computed expectations for R1 (suppress_spawns), R2 (body_count),
R3 (speed_aware_radius), and V1 parity when all three are ablated off. See
scratchpad/gauntlet/L67/opp_fix/O13_v2.md for the data-derived constants (R_SPAWN, W, R_BODY) and the
SPAWNS/BODIES verification against pipeline.vocab / CardDB.

NOTE on R3's bandit scenario: real ``CardDB.speed_tiles('bandit') == 1.5`` does NOT, by the ticket's own
formula/constants (K=1.5, cap=0.25), raise the effective radius above 0.07 -- MEASURED in O13_v2.md, and
arithmetically incapable of covering the ticket's own 0.12 scenario. The test below uses a stub speed value
large enough to demonstrate the R3 MECHANISM (matching, radius widening, ablation), not real bandit data;
that finding is not silently worked around.
"""
from __future__ import annotations

import unittest

from clashrl.opponent_elixir import (
    BODIES, SPAWNS, V2_PARAMS, OpponentElixirEstimator, OpponentElixirEstimatorV2,
)


class _FakeDB:
    """Same stand-in convention as icebow/tests/test_opponent_elixir.py, extended with `speed_tiles` for R3."""

    def __init__(self, costs, speeds=None):
        self._costs = dict(costs)
        self._speeds = dict(speeds or {})

    def elixir(self, base):
        return self._costs.get(base)

    def speed_tiles(self, base):
        return self._speeds.get(base)


class _Det:
    def __init__(self, base, cx, gy, team="enemy"):
        self.base = base
        self.cx = cx
        self.gy = gy
        self.team = team


# ------------------------------------------------------------------------------------------------------
# SPAWNS / BODIES table sanity -- every class name must be a real vocab class (never asserted against
# vocab here to avoid a circular import assumption; pipeline's own test suite cross-checks against
# pipeline.vocab -- see test_opp_est_audit.py's TestV2Params). This just checks the tables are internally
# consistent (no empty sets, all string keys/members) and match the ticket's corrections (O13_v2.md).
# ------------------------------------------------------------------------------------------------------
class TestSpawnsBodiesTables(unittest.TestCase):
    def test_every_spawner_maps_to_itself(self):
        for parent, spawns in SPAWNS.items():
            self.assertIn(parent, spawns, f"{parent} must map to itself (engine self-naming)")

    def test_corrections_applied(self):
        # ticket's "fire_spirits"/"cursed_hog"/"lava_pup"(singular)/"phoenix" were dropped or corrected --
        # see O13_v2.md for why each is not a real vocab class.
        self.assertNotIn("fire_spirits", SPAWNS.get("furnace", ()))
        self.assertIn("fire_spirit", SPAWNS.get("furnace", ()))
        self.assertNotIn("cursed_hog", SPAWNS.get("mother_witch", ()))
        self.assertNotIn("hog", SPAWNS.get("mother_witch", ()))
        self.assertIn("mother_witch_hog", SPAWNS.get("mother_witch", ()))
        self.assertNotIn("lava_pup", SPAWNS.get("lava_hound", ()))
        self.assertIn("lava_pups", SPAWNS.get("lava_hound", ()))
        self.assertNotIn("phoenix", SPAWNS)
        self.assertNotIn("fire_spirits", BODIES)
        self.assertNotIn("cursed_hog", BODIES)

    def test_lava_pups_body_count_from_spawner_on_death(self):
        # ticket says 6; verified against CardDB.spawner('lava_hound')['on_death'], not the lava_pups card's
        # own 'count' field (which describes one pup, not the death-spawn burst) -- O13_v2.md.
        self.assertEqual(BODIES["lava_pups"], 6)


# ------------------------------------------------------------------------------------------------------
# V1 parity golden capture (BEFORE this ticket's edit) -- OpponentElixirEstimator itself must be untouched.
# Captured by loading the git-committed HEAD (64d7eca) copy of opponent_elixir.py as a separate module, so
# this test can never pass by coincidence if the live import were accidentally modified.
# ------------------------------------------------------------------------------------------------------
def _load_golden_v1():
    """O13 attempt 2 FIX 5 (verifier, LOW): the git-committed HEAD-64d7eca source is still captured with
    ``git show`` at test time (that part was right), but is now written under the OS temp dir (via
    ``tempfile``), never into the repo tree -- the previous version left a stray
    ``scratchpad/gauntlet/L67/opp_fix/_golden_v1_64d7eca.py`` file behind after every test run, which is not
    a backup or a progress artifact and does not belong in the write set."""
    import importlib.util
    import subprocess
    import tempfile
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    src = subprocess.check_output(
        ["git", "show", "64d7eca:icebow/src/clashrl/opponent_elixir.py"], cwd=str(repo)
    ).decode("utf-8")
    fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="golden_opponent_elixir_v1_64d7eca_")
    tmp = Path(tmp_path)
    try:
        with open(fd, "w", encoding="utf-8") as fh:
            fh.write(src)
        spec = importlib.util.spec_from_file_location("_golden_opponent_elixir_v1", tmp)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.OpponentElixirEstimator
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


# One-time cleanup: an earlier version of this test wrote its golden copy into the repo's scratchpad tree
# instead of a temp file (FIX 5 above) -- remove that stray artifact if a prior run left it behind.
def _remove_stray_golden_in_repo_tree() -> None:
    from pathlib import Path

    stray = (Path(__file__).resolve().parents[2] / "scratchpad" / "gauntlet" / "L67" / "opp_fix"
            / "_golden_v1_64d7eca.py")
    if stray.exists():
        stray.unlink()


_remove_stray_golden_in_repo_tree()


class TestV1ParityUnchanged(unittest.TestCase):
    """The live OpponentElixirEstimator must behave identically to the HEAD-64d7eca golden copy on 3
    synthetic sequences -- the class this ticket must not alter."""

    @classmethod
    def setUpClass(cls):
        cls.Golden = _load_golden_v1()

    def _run(self, cls, db, seq):
        est = cls(db)
        est.reset(my_elixir=5.0, now=0.0)
        out = []
        for my_elixir, dets, now, my_play in seq:
            if my_play:
                est.record_my_play(my_play)
            out.append(round(est.update(my_elixir, dets, now), 6))
        return out

    def test_sequence_1_simple_plays(self):
        db = _FakeDB({"ice_wizard": 3, "skeletons": 1, "knight": 3})
        seq = [
            (5.0, [], 0.0, "ice_wizard"),
            (5.0, [_Det("skeletons", 0.5, 0.5)], 1.0, None),
            (5.0, [_Det("skeletons", 0.51, 0.5)], 2.0, None),
            (6.0, [_Det("knight", 0.2, 0.2)], 3.0, None),
        ]
        self.assertEqual(self._run(OpponentElixirEstimator, db, seq), self._run(self.Golden, db, seq))

    def test_sequence_2_saturation(self):
        db = _FakeDB({"golem": 8, "pekka": 7})
        seq = [
            (10.0, [_Det("golem", 0.5, 0.5)], 0.0, None),
            (10.0, [_Det("pekka", 0.3, 0.3)], 0.5, None),
            (10.0, [], 1.0, None),
        ]
        self.assertEqual(self._run(OpponentElixirEstimator, db, seq), self._run(self.Golden, db, seq))

    def test_sequence_3_expiry_and_rebill(self):
        db = _FakeDB({"miner": 3})
        seq = [
            (5.0, [_Det("miner", 0.5, 0.5)], 0.0, None),
            (5.0, [], 7.0, None),                              # track expires (forget_s=6.0)
            (5.0, [_Det("miner", 0.5, 0.5)], 7.1, None),        # rebilled -- V1's own known bug, unchanged
        ]
        self.assertEqual(self._run(OpponentElixirEstimator, db, seq), self._run(self.Golden, db, seq))


# ------------------------------------------------------------------------------------------------------
# R1 -- suppress_spawns
# ------------------------------------------------------------------------------------------------------
class TestR1SuppressSpawns(unittest.TestCase):
    def test_live_form_skeletons_near_tombstone_not_charged(self):
        db = _FakeDB({"tombstone": 3, "skeletons": 1})
        est = OpponentElixirEstimatorV2(db)
        est.reset(my_elixir=5.0, now=0.0)
        est.update(5.0, [_Det("tombstone", 0.50, 0.50)], now=0.0)
        self.assertEqual(est._opp_spent, 3.0)
        est.update(5.0, [_Det("tombstone", 0.50, 0.50), _Det("skeletons", 0.55, 0.50)], now=1.0)
        self.assertEqual(est._opp_spent, 3.0, "live-form spawn (different class) near tombstone must not charge")

    def test_engine_form_second_tombstone_label_near_tombstone_not_charged(self):
        db = _FakeDB({"tombstone": 3, "skeletons": 1})
        est = OpponentElixirEstimatorV2(db)
        est.reset(my_elixir=5.0, now=0.0)
        est.update(5.0, [_Det("tombstone", 0.50, 0.50)], now=0.0)
        est.update(5.0, [_Det("tombstone", 0.50, 0.50), _Det("tombstone", 0.55, 0.50)], now=1.0)
        self.assertEqual(est._opp_spent, 3.0, "engine-form spawn (same class, self-mapped) must not charge")

    def test_skeletons_far_from_spawner_is_charged(self):
        db = _FakeDB({"tombstone": 3, "skeletons": 1})
        est = OpponentElixirEstimatorV2(db)
        est.reset(my_elixir=5.0, now=0.0)
        est.update(5.0, [_Det("tombstone", 0.50, 0.50)], now=0.0)
        est.update(5.0, [_Det("tombstone", 0.50, 0.50), _Det("skeletons", 1.00, 0.50)], now=1.0)
        self.assertEqual(est._opp_spent, 4.0, "a genuinely unrelated skeletons play, far away, must charge")

    def test_ablation_suppress_spawns_off_charges_the_spawn(self):
        db = _FakeDB({"tombstone": 3, "skeletons": 1})
        est = OpponentElixirEstimatorV2(db, suppress_spawns=False)
        est.reset(my_elixir=5.0, now=0.0)
        est.update(5.0, [_Det("tombstone", 0.50, 0.50)], now=0.0)
        est.update(5.0, [_Det("tombstone", 0.50, 0.50), _Det("skeletons", 0.55, 0.50)], now=1.0)
        self.assertEqual(est._opp_spent, 4.0, "rule OFF must reproduce V1's over-billing for this scenario")


# ------------------------------------------------------------------------------------------------------
# R2 -- body_count
# ------------------------------------------------------------------------------------------------------
class TestR2BodyCount(unittest.TestCase):
    def test_three_skeletons_one_update_spread_beyond_cluster_radius_one_charge(self):
        db = _FakeDB({"skeletons": 1})
        est = OpponentElixirEstimatorV2(db)
        est.reset(my_elixir=5.0, now=0.0)
        est.update(5.0, [_Det("skeletons", 0.30, 0.5), _Det("skeletons", 0.45, 0.5),
                        _Det("skeletons", 0.60, 0.5)], now=0.0)   # 0.15 apart, > cluster_radius (0.10)
        self.assertEqual(est._opp_spent, 1.0)
        self.assertEqual(len(est._tracks), 3, "all 3 bodies still become tracks for future matching")

    def test_fifteen_skeleton_army_over_two_updates_absorbed_one_charge(self):
        db = _FakeDB({"skeleton_army": 3})
        est = OpponentElixirEstimatorV2(db)
        est.reset(my_elixir=10.0, now=0.0)
        wave1 = [_Det("skeleton_army", 0.50 + 0.005 * i, 0.5) for i in range(8)]
        wave2 = [_Det("skeleton_army", 0.50 + 0.005 * i, 0.5) for i in range(8, 15)]
        est.update(10.0, wave1, now=0.0)
        self.assertEqual(est._opp_spent, 3.0)
        est.update(10.0, wave1 + wave2, now=0.25)          # +7 bodies, same spot, within W=1.0s
        self.assertEqual(est._opp_spent, 3.0, "second wave absorbed into the same deployment, no new charge")

    def test_two_plays_five_seconds_apart_different_spot_two_charges(self):
        """Body-group window (W=1.0s) isolated from track persistence: a DIFFERENT nearby spot so the
        lingering forget_s=6.0s track from play 1 cannot MATCH play 2's dets by proximity -- this is the
        scenario R2's body-group window (not V1's forget_s) is responsible for charging correctly."""
        db = _FakeDB({"skeletons": 1})
        est = OpponentElixirEstimatorV2(db)
        est.reset(my_elixir=10.0, now=0.0)
        est.update(10.0, [_Det("skeletons", 0.30, 0.5), _Det("skeletons", 0.32, 0.5),
                         _Det("skeletons", 0.34, 0.5)], now=0.0)
        self.assertEqual(est._opp_spent, 1.0)
        est.update(10.0, [_Det("skeletons", 0.70, 0.5), _Det("skeletons", 0.72, 0.5),
                         _Det("skeletons", 0.74, 0.5)], now=5.0)
        self.assertEqual(est._opp_spent, 2.0)

    def test_two_plays_same_spot_within_forget_s_absorbed_one_charge_KNOWN_LIMITATION(self):
        """KNOWN LIMITATION (O13 attempt 2 FIX 4, verifier LOW), inherited from V1 unmodified, NOT something
        R2 introduces or could fix without changing forget_s (out of scope, ticket: "do NOT change forget_s"):
        a second play of the SAME card at the literal SAME spot, within forget_s=6.0s of the first, is
        matched by ``_find_track`` to the still-live track from play 1 (V1's own track-matching, R3 off since
        neither is dashing) and REFRESHES it instead of creating new tracks -- so R2's body-group logic never
        even runs (there are no new tracks to group), and the second play is silently absorbed as if it were
        the same units still standing there. This is a pre-existing V1 behavior this ticket does not touch;
        it is the ORIGINAL scenario text ("two skeletons plays 5 s apart at the same spot -> TWO charges")
        that was wrong about V1-inherited track persistence, not a V2 R2 bug. See
        test_two_plays_five_seconds_apart_different_spot_two_charges above for R2's actual behavior, and
        test_two_plays_same_spot_after_forget_s_two_charges below for the literal same-spot case once the
        track has actually expired."""
        db = _FakeDB({"skeletons": 1})
        est = OpponentElixirEstimatorV2(db)
        est.reset(my_elixir=10.0, now=0.0)
        est.update(10.0, [_Det("skeletons", 0.30, 0.5), _Det("skeletons", 0.32, 0.5),
                         _Det("skeletons", 0.34, 0.5)], now=0.0)
        self.assertEqual(est._opp_spent, 1.0)
        est.update(10.0, [_Det("skeletons", 0.30, 0.5), _Det("skeletons", 0.32, 0.5),
                         _Det("skeletons", 0.34, 0.5)], now=5.0)   # same spot, 5s later (< forget_s=6.0)
        self.assertEqual(est._opp_spent, 1.0, "known limitation: absorbed by the still-live V1 track")
        # play 1's 3 dets are within cluster_radius (0.10) of each other, so they merge into ONE track
        # (n=3, centroid) even on the FIRST play -- play 2's dets all re-match that same single track, so
        # no NEW track is created either time.
        self.assertEqual(len(est._tracks), 1, "no NEW track was created -- all 3 dets matched/refreshed")

    def test_two_plays_same_spot_after_forget_s_two_charges(self):
        """Same spot as above, but 7s apart (> forget_s=6.0s): the first play's tracks have actually
        EXPIRED by the second play, so its dets are genuinely new -> TWO charges, as the original ticket
        scenario intended once the confound above is accounted for."""
        db = _FakeDB({"skeletons": 1})
        est = OpponentElixirEstimatorV2(db)
        est.reset(my_elixir=10.0, now=0.0)
        est.update(10.0, [_Det("skeletons", 0.30, 0.5), _Det("skeletons", 0.32, 0.5),
                         _Det("skeletons", 0.34, 0.5)], now=0.0)
        self.assertEqual(est._opp_spent, 1.0)
        est.update(10.0, [_Det("skeletons", 0.30, 0.5), _Det("skeletons", 0.32, 0.5),
                         _Det("skeletons", 0.34, 0.5)], now=7.0)   # same spot, 7s later (> forget_s=6.0)
        self.assertEqual(est._opp_spent, 2.0)

    def test_ablation_body_count_off_charges_once_per_cluster(self):
        db = _FakeDB({"skeletons": 1})
        est = OpponentElixirEstimatorV2(db, body_count=False)
        est.reset(my_elixir=5.0, now=0.0)
        est.update(5.0, [_Det("skeletons", 0.30, 0.5), _Det("skeletons", 0.45, 0.5),
                        _Det("skeletons", 0.60, 0.5)], now=0.0)
        self.assertEqual(est._opp_spent, 3.0, "rule OFF must reproduce V1's one-charge-per-cluster over-billing")


# ------------------------------------------------------------------------------------------------------
# R3 -- speed_aware_radius (bandit stub speed -- see module docstring for why real cards.yaml data can't
# demonstrate this; O13_v2.md has the measured gap)
# ------------------------------------------------------------------------------------------------------
class TestR3SpeedAwareRadius(unittest.TestCase):
    def test_bandit_dash_matched_no_charge(self):
        # stub speed chosen so speed_norm*dt*K covers the 0.12 move: 0.12/(0.25*1.5)=0.32/s -> speed_tiles
        # >= 0.32*18=5.76; 6.0 used with headroom. NOT bandit's real cards.yaml speed_tiles (1.5) -- see
        # module docstring / O13_v2.md.
        db = _FakeDB({"bandit": 3}, speeds={"bandit": 6.0})
        est = OpponentElixirEstimatorV2(db)
        est.reset(my_elixir=5.0, now=0.0)
        est.update(5.0, [_Det("bandit", 0.50, 0.5)], now=0.0)
        self.assertEqual(est._opp_spent, 3.0)
        est.update(5.0, [_Det("bandit", 0.62, 0.5)], now=0.25)     # moved 0.12 in 0.25s
        self.assertEqual(est._opp_spent, 3.0, "fast/dashing unit must match its track, not rebill")
        self.assertEqual(len(est._tracks), 1, "must be one continuing track, not two")

    def test_stationary_tombstone_radius_unchanged(self):
        db = _FakeDB({"tombstone": 3}, speeds={})       # no speed data for tombstone (a building)
        est = OpponentElixirEstimatorV2(db)
        est.reset(my_elixir=5.0, now=0.0)
        est.update(5.0, [_Det("tombstone", 0.50, 0.50)], now=0.0)
        est.update(5.0, [_Det("tombstone", 0.505, 0.50)], now=0.25)   # 0.005 move, well within match_radius
        self.assertEqual(est._opp_spent, 3.0)
        self.assertEqual(len(est._tracks), 1)

    def test_ablation_speed_aware_off_rebills_the_dash(self):
        # Charge-count check, not a raw opp_spent number: at my_elixir=5.0 a second 3-cost charge pushes the
        # books past the [0,10] cap, and the L67g saturation rebase (copied verbatim from V1) claws back the
        # overflow into opp_spent -- that correction is real V1 behaviour, not a defect in this ablation
        # check. `_body_groups`/track count is the unambiguous signal that a SECOND, separate charge event
        # fired (a real match would never hold my_elixir fixed at 5.0 across two live samples anyway).
        db = _FakeDB({"bandit": 3}, speeds={"bandit": 6.0})
        est = OpponentElixirEstimatorV2(db, speed_aware_radius=False)
        est.reset(my_elixir=5.0, now=0.0)
        est.update(5.0, [_Det("bandit", 0.50, 0.5)], now=0.0)
        self.assertEqual(est._opp_spent, 3.0)
        est.update(5.0, [_Det("bandit", 0.62, 0.5)], now=0.25)
        self.assertEqual(len(est._tracks), 2, "rule OFF must reproduce V1's fixed-radius rebill (2nd track)")
        self.assertGreater(est._opp_spent, 3.0, "a second charge must have been billed")


# ------------------------------------------------------------------------------------------------------
# All three rules off => byte-for-byte same tracking/charging decisions as V1 (same match_radius,
# cluster_radius, forget_s, one-charge-per-cluster, no suppression).
# ------------------------------------------------------------------------------------------------------
class TestAblationAllOffMatchesV1(unittest.TestCase):
    def _run(self, cls, db, seq, **kwargs):
        est = cls(db, **kwargs) if kwargs else cls(db)
        est.reset(my_elixir=5.0, now=0.0)
        out = []
        for my_elixir, dets, now, my_play in seq:
            if my_play:
                est.record_my_play(my_play)
            out.append(round(est.update(my_elixir, dets, now), 6))
        return out, est._opp_spent, len(est._tracks)

    def test_matches_v1_on_a_mixed_scenario(self):
        db = _FakeDB({"tombstone": 3, "skeletons": 1, "bandit": 3, "skeleton_army": 3},
                     speeds={"bandit": 6.0})
        seq = [
            (5.0, [_Det("tombstone", 0.50, 0.50)], 0.0, None),
            (5.0, [_Det("tombstone", 0.50, 0.50), _Det("skeletons", 0.55, 0.50)], 1.0, None),
            (5.0, [_Det("bandit", 0.20, 0.20)], 2.0, None),
            (5.0, [_Det("bandit", 0.32, 0.20)], 2.25, None),
            (5.0, [_Det("skeleton_army", 0.10 + 0.15 * i, 0.10) for i in range(3)], 3.0, None),
        ]
        v1_out, v1_spent, v1_ntracks = self._run(OpponentElixirEstimator, db, seq)
        v2_out, v2_spent, v2_ntracks = self._run(
            OpponentElixirEstimatorV2, db, seq,
            suppress_spawns=False, body_count=False, speed_aware_radius=False,
        )
        self.assertEqual(v1_out, v2_out)
        self.assertEqual(v1_spent, v2_spent)
        self.assertEqual(v1_ntracks, v2_ntracks)


# ------------------------------------------------------------------------------------------------------
# V2_PARAMS sanity -- every constant this class's DEFAULTS actually use must be logged (ticket 3).
# ------------------------------------------------------------------------------------------------------
class TestV2ParamsLogged(unittest.TestCase):
    def test_defaults_match_v2_params(self):
        db = _FakeDB({})
        est = OpponentElixirEstimatorV2(db)
        self.assertEqual(est.r_spawn, V2_PARAMS["r_spawn"])
        self.assertEqual(est.r_body, V2_PARAMS["r_body"])
        self.assertEqual(est.body_group_window_s, V2_PARAMS["body_group_window_s"])
        self.assertEqual(est.speed_k, V2_PARAMS["speed_k"])
        self.assertEqual(est.speed_radius_cap, V2_PARAMS["speed_radius_cap"])
        self.assertEqual(est.match_radius, V2_PARAMS["match_radius"])
        self.assertEqual(est.cluster_radius, V2_PARAMS["cluster_radius"])
        self.assertEqual(est.forget_s, V2_PARAMS["forget_s"])
        self.assertIs(est.spawns, SPAWNS)
        self.assertIs(est.bodies, BODIES)


if __name__ == "__main__":
    unittest.main()
