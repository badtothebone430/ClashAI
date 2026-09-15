"""Tests for the E1 attribution screen: e1_view.Noise's per-component switches over live_view, and
e1_eval's ``--noise-off`` CLI (HANDOFF "AW. L67aq" proposal 1: "split the 40-point gap by noise source").

    icebow/.venv/Scripts/python.exe -m unittest pipeline.tests.test_e1_noise_arms -v

No engine, no GPU, no training, no network. ``patch("pipeline.e1_view.DEGRADE_RECALL", ...)`` etc. below only
rebinds the NAME e1_view imported into its own module for the duration of one test -- obs_contract.py itself
is never touched.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline import e1_eval                                  # noqa: E402
from pipeline import e1_view                                   # noqa: E402
from pipeline import obs_contract as oc                        # noqa: E402
from pipeline import vocab                                     # noqa: E402
from pipeline.e1_view import ALL_NOISE_OFF, Noise, live_view    # noqa: E402
from pipeline.obs_contract import UNKNOWN_TEAM_RATE             # noqa: E402
from pipeline.tests.test_obs_contract import ENGINE_DECK, raw_obs   # noqa: E402

NOISE_FIELDS = ("recall", "false_pos", "position", "team", "unit_hp", "scalars", "deploying", "conf")


def _is_subsequence(sub, full) -> bool:
    """True iff every item of ``sub`` occurs in ``full``, in the same order (not necessarily contiguous)."""
    it = iter(full)
    return all(any(x == y for y in it) for x in sub)


def _extra_after_subsequence(sub, full) -> list:
    """Assuming ``sub`` is a subsequence of ``full`` (checked by the caller via ``_is_subsequence``), return
    the elements of ``full`` that are NOT part of that match -- e.g. the rescued (recall=False) units, given
    ``sub`` = the default (recall=True) path's kept units."""
    extra = []
    it = iter(sub)
    target = next(it, None)
    for item in full:
        if target is not None and item == target:
            target = next(it, None)
        else:
            extra.append(item)
    return extra


def _deck():
    return oc.load_deck("icebow")


def _bs(deck, *, extra_enemy: int = 6) -> oc.BoardState:
    """The obs_contract fixture board plus extra enemy bodies icebow cannot produce, through the SAME
    compact_raw -> from_engine path e1_eval uses (matches test_e1_baseline._engine_bs)."""
    from pipeline import engine_play as ep
    o = raw_obs(0)
    names = ["HogRider", "Musketeer", "Valkyrie", "Giant", "MiniPekka", "Wizard"]
    for i in range(extra_enemy):
        o["entities"].append({"side": 1, "x": 3000 + 2000 * i, "y": 18000 + 500 * i, "card_id": 26000100 + i,
                              "name": names[i % len(names)], "hp": 500, "max_hp": 1000, "kind": 15})
    return oc.from_engine(ep.compact_raw(o), 0, deck, engine_deck=ENGINE_DECK, unmapped=set())


def _reference_live_view(bs: oc.BoardState, rng: np.random.Generator, deck: oc.Deck) -> oc.BoardState:
    """The PRE-Noise live_view, written independently of e1_view.py, straight from its own module docstring:
    obs_contract.degrade() (untouched, all defaults) + the three live fill rules. Used to pin item 2 against
    the unmodified obs_contract.py rather than against e1_view's own (now Noise-aware) implementation."""
    d = oc.degrade(bs, rng)
    allowed = oc.mine_classes(deck)

    def fill(u):
        side = u.side
        if side < 0 and vocab.base_key(vocab.UNIT_VOCAB[int(u.cls)]) not in allowed:
            side = 1
        return replace(u, hp_frac=1.0, side=side)

    units = tuple(fill(u) for u in d.units)
    spells = tuple(fill(u) for u in d.spells)
    towers = tuple(replace(t, hp_frac=1.0) if (t.kind == "king" and t.alive) else t for t in d.towers)
    return replace(d, units=units, spells=spells, towers=towers)


# ------------------------------------------------------------------------------------------------------
# item 2: all switches at defaults == today's live_view / obs_contract.degrade, bit-identical
# ------------------------------------------------------------------------------------------------------
class TestAllOnBitIdentical(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = _deck()
        cls.boards = [_bs(cls.deck), _bs(cls.deck, extra_enemy=0), raw_obs_board(cls.deck)]

    def test_degrade_switchable_matches_obs_contract_degrade(self):
        for bi, bs in enumerate(self.boards):
            for seed in range(12):
                d_ref = oc.degrade(bs, np.random.default_rng(seed))
                d_new = e1_view._degrade_switchable(bs, np.random.default_rng(seed), Noise())
                self.assertEqual(d_ref, d_new, (bi, seed))

    def test_live_view_matches_hand_written_reference(self):
        for bi, bs in enumerate(self.boards):
            for seed in range(12):
                ref = _reference_live_view(bs, np.random.default_rng(seed), self.deck)
                new = live_view(bs, np.random.default_rng(seed), self.deck)                  # default Noise
                new_explicit = live_view(bs, np.random.default_rng(seed), self.deck, Noise())
                self.assertEqual(ref, new, (bi, seed))
                self.assertEqual(ref, new_explicit, (bi, seed))
                tok_r, mask_r, sc_r = oc.to_tokens(ref, 64)
                tok_n, mask_n, sc_n = oc.to_tokens(new, 64)
                self.assertTrue(np.array_equal(tok_r, tok_n) and np.array_equal(mask_r, mask_n)
                                and np.array_equal(sc_r, sc_n))


def raw_obs_board(deck):
    from pipeline import engine_play as ep
    return oc.from_engine(ep.compact_raw(raw_obs(1)), 1, deck, engine_deck=ENGINE_DECK, unmapped=set())


# ------------------------------------------------------------------------------------------------------
# item 4: all switches off == the clean path (--obs clean), tokens/scalars
# ------------------------------------------------------------------------------------------------------
class TestAllOffMatchesClean(unittest.TestCase):
    def test_tokens_and_scalars_match_clean(self):
        deck = _deck()
        bs = _bs(deck)
        for seed in range(10):
            v = live_view(bs, np.random.default_rng(seed), deck, ALL_NOISE_OFF)
            tok_v, mask_v, sc_v = oc.to_tokens(v, 64)
            tok_c, mask_c, sc_c = oc.to_tokens(bs, 64)
            self.assertTrue(np.array_equal(tok_v, tok_c), seed)
            self.assertTrue(np.array_equal(mask_v, mask_c), seed)
            self.assertTrue(np.array_equal(sc_v, sc_c), seed)
        # documented, intentional difference: metadata not read by to_tokens (module docstring / Noise docstring)
        v = live_view(bs, np.random.default_rng(0), deck, ALL_NOISE_OFF)
        self.assertEqual(v.source, "degraded")
        self.assertEqual(bs.source, "engine")
        self.assertEqual(v.t_source, "clock")
        self.assertEqual(bs.t_source, "tick")

    def test_deploying_off_keeps_true_value_even_though_compact_rows_are_none(self):
        deck = _deck()
        base = _bs(deck)
        bs = replace(base, units=(replace(base.units[0], deploying=True),) + base.units[1:])
        v = live_view(bs, np.random.default_rng(0), deck, ALL_NOISE_OFF)
        self.assertTrue(v.units[0].deploying)          # kept true, not dropped to None
        v_on = live_view(bs, np.random.default_rng(0), deck)     # default: deploying dropped
        self.assertIsNone(v_on.units[0].deploying)


# ------------------------------------------------------------------------------------------------------
# item 3: RNG stability -- one component off changes only its own effect
# ------------------------------------------------------------------------------------------------------
class TestRngStability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = _deck()
        cls.bs = _bs(cls.deck)          # 3 mine + 3 real enemy + 6 extra unmapped-class enemy = enough units
        # a unit with true deploying=True and non-trivial hp, to make unit_hp/deploying assertions non-trivial
        cls.bs = replace(cls.bs, units=(replace(cls.bs.units[0], deploying=True),) + cls.bs.units[1:])

    def test_field_only_components_do_not_change_which_units_or_fps_appear(self):
        """position / team / conf / unit_hp / scalars / deploying never gate degrade()'s control flow: with
        one of them off, the SAME units (by class, in the SAME order) are kept and the SAME false positives
        are added, for every seed -- only that switch's own field(s) may differ."""
        field_only = ("position", "team", "conf", "unit_hp", "scalars", "deploying")
        counts_seen = set()
        for seed in range(60):
            d0 = e1_view._degrade_switchable(self.bs, np.random.default_rng(seed), Noise())
            counts_seen.add(len(d0.units))
            for name in field_only:
                d1 = e1_view._degrade_switchable(self.bs, np.random.default_rng(seed), Noise(**{name: False}))
                self.assertEqual([u.cls for u in d0.units], [u.cls for u in d1.units], (name, seed))
                self.assertEqual(len(d0.units), len(d1.units), (name, seed))
        self.assertGreater(len(counts_seen), 1, "recall/false_pos never varied the unit count over 60 seeds")

    def test_position_off_only_changes_xy(self):
        for seed in range(30):
            d0 = e1_view._degrade_switchable(self.bs, np.random.default_rng(seed), Noise())
            d1 = e1_view._degrade_switchable(self.bs, np.random.default_rng(seed), Noise(position=False))
            for u0, u1 in zip(d0.units, d1.units):
                self.assertEqual((u0.cls, u0.side, u0.hp_frac, u0.deploying, u0.conf),
                                 (u1.cls, u1.side, u1.hp_frac, u1.deploying, u1.conf), seed)

    def test_team_off_keeps_true_side_and_no_other_field_changes(self):
        for seed in range(30):
            d0 = e1_view._degrade_switchable(self.bs, np.random.default_rng(seed), Noise())
            d1 = e1_view._degrade_switchable(self.bs, np.random.default_rng(seed), Noise(team=False))
            for u0, u1 in zip(d0.units, d1.units):
                self.assertEqual((u0.cls, u0.x, u0.y, u0.hp_frac, u0.deploying, u0.conf),
                                 (u1.cls, u1.x, u1.y, u1.hp_frac, u1.deploying, u1.conf), seed)
                self.assertIn(u1.side, (0, 1))          # never -1: no unknown-team roll applied

    def test_conf_off_keeps_true_conf(self):
        for seed in range(30):
            d0 = e1_view._degrade_switchable(self.bs, np.random.default_rng(seed), Noise())
            d1 = e1_view._degrade_switchable(self.bs, np.random.default_rng(seed), Noise(conf=False))
            for u0, u1 in zip(d0.units, d1.units):
                self.assertEqual((u0.cls, u0.x, u0.y, u0.side, u0.hp_frac, u0.deploying),
                                 (u1.cls, u1.x, u1.y, u1.side, u1.hp_frac, u1.deploying), seed)
                self.assertEqual(u1.conf, 1.0)          # engine truth conf, never redrawn

    @patch("pipeline.e1_view.DEGRADE_RECALL", 1.0)
    @patch("pipeline.e1_view.DEGRADE_PRECISION", 1.0)
    def test_field_components_reduce_to_true_engine_values(self):
        """With drop/false-positive both forced off (patched, obs_contract.py untouched), each field-only
        switch off reproduces the exact engine truth for that field, unit for unit."""
        d = e1_view._degrade_switchable(self.bs, np.random.default_rng(0), Noise(position=False))
        for u, src in zip(d.units, self.bs.units):
            self.assertEqual((u.x, u.y), (src.x, src.y))
        d = e1_view._degrade_switchable(self.bs, np.random.default_rng(0), Noise(team=False))
        for u, src in zip(d.units, self.bs.units):
            self.assertEqual(u.side, src.side)
        d = e1_view._degrade_switchable(self.bs, np.random.default_rng(0), Noise(conf=False))
        for u, src in zip(d.units, self.bs.units):
            self.assertEqual(u.conf, src.conf)
        d = e1_view._degrade_switchable(self.bs, np.random.default_rng(0), Noise(unit_hp=False))
        for u, src in zip(d.units, self.bs.units):
            self.assertEqual(u.hp_frac, src.hp_frac)
        d = e1_view._degrade_switchable(self.bs, np.random.default_rng(0), Noise(deploying=False))
        for u, src in zip(d.units, self.bs.units):
            self.assertEqual(u.deploying, src.deploying)
        d = e1_view._degrade_switchable(self.bs, np.random.default_rng(0), Noise(scalars=False))
        self.assertEqual(d.my_elixir, self.bs.my_elixir)
        self.assertTrue(d.my_elixir_exact)
        self.assertEqual(d.opp_elixir, self.bs.opp_elixir)

    @patch("pipeline.e1_view.DEGRADE_PRECISION", 1.0)
    def test_recall_off_drops_nothing_and_keeps_the_kept_units_untouched(self):
        """precision patched to 1.0 (fp_rate 0, obs_contract.py untouched) isolates recall: OFF must equal
        the full board every time, and every unit degrade() itself would have kept is bit-identical to the
        default run (no extra draws are spent noising the rescued ones)."""
        n_units = len(self.bs.units)
        drops_seen = kept_seen = 0
        for seed in range(80):
            d_on = e1_view._degrade_switchable(self.bs, np.random.default_rng(seed), Noise())
            d_off = e1_view._degrade_switchable(self.bs, np.random.default_rng(seed), Noise(recall=False))
            self.assertEqual(len(d_off.units), n_units, seed)          # nothing ever dropped
            if len(d_on.units) < n_units:
                drops_seen += 1
            else:
                kept_seen += 1
                self.assertEqual(d_on.units, d_off.units, seed)        # nothing was rescued -> identical
        self.assertGreater(drops_seen, 0, "DEGRADE_RECALL=0.855 never dropped a unit over 80 seeds")
        self.assertGreater(kept_seen, 0)

    @patch("pipeline.e1_view.DEGRADE_RECALL", 1.0)
    def test_false_pos_off_adds_nothing_and_keeps_the_real_units_untouched(self):
        """recall patched to 1.0 (nothing dropped, obs_contract.py untouched) isolates false_pos: OFF must
        equal exactly the real units every time, and match the default run's real units field for field."""
        n_units = len(self.bs.units)
        fp_seen = clean_seen = 0
        for seed in range(80):
            d_on = e1_view._degrade_switchable(self.bs, np.random.default_rng(seed), Noise())
            d_off = e1_view._degrade_switchable(self.bs, np.random.default_rng(seed), Noise(false_pos=False))
            self.assertEqual(len(d_off.units), n_units, seed)          # never a false positive
            self.assertTrue(_is_subsequence(d_off.units, d_on.units), seed)  # the real units, untouched,
            # in order -- NOT a fixed prefix, since an earlier false positive shifts later ones' index
            if len(d_on.units) > n_units:
                fp_seen += 1
            else:
                clean_seen += 1
        self.assertGreater(fp_seen, 0, "DEGRADE_PRECISION=0.886 never added a false positive over 80 seeds")
        self.assertGreater(clean_seen, 0)


# ------------------------------------------------------------------------------------------------------
# O1 repair: recall=False must NOISE the rescued units (position/team/hp/deploying/conf), not append them
# clean -- an independent "side" RNG stream carries that noise so the main stream is untouched either way.
# ------------------------------------------------------------------------------------------------------
class TestRecallRescueNoise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deck = _deck()
        cls.bs = _bs(cls.deck, extra_enemy=0)      # 5 units, each a UNIQUE class -> unambiguous cls lookup

    def test_side_rng_creation_does_not_advance_main_rng(self):
        """``Generator(rng.bit_generator.jumped()).random()`` must not change what ``rng`` draws next."""
        rng = np.random.default_rng(123)
        before = [rng.random() for _ in range(4)]
        _side = np.random.Generator(rng.bit_generator.jumped())
        _ = _side.random(3)                                     # drawing from the side stream too: still inert
        after = [rng.random() for _ in range(4)]
        ref = np.random.default_rng(123)
        expected = [ref.random() for _ in range(8)]
        self.assertEqual(before + after, expected)

    @patch("pipeline.e1_view.DEGRADE_PRECISION", 1.0)   # isolate recall: no FPs, so d_on is a clean subsequence
    def test_kept_units_identical_to_default_rescued_units_are_extra(self):
        """recall=False's KEPT units (i.e. everything degrade() itself would also have kept) are bit-identical
        to the default path's, in the same order -- rescued units are additional entries, never replacements."""
        for seed in range(60):
            d_on = e1_view._degrade_switchable(self.bs, np.random.default_rng(seed), Noise())
            d_off = e1_view._degrade_switchable(self.bs, np.random.default_rng(seed), Noise(recall=False))
            self.assertEqual(len(d_off.units), len(self.bs.units), seed)     # recall off drops nothing
            self.assertTrue(_is_subsequence(d_on.units, d_off.units), seed)

    @patch("pipeline.e1_view.DEGRADE_PRECISION", 1.0)
    def test_rescued_units_are_noised_like_kept_ones(self):
        by_cls = {u.cls: u for u in self.bs.units}
        dxdy: list[float] = []
        unknown_side = total = 0
        for seed in range(300):
            d_on = e1_view._degrade_switchable(self.bs, np.random.default_rng(seed), Noise())
            d_off = e1_view._degrade_switchable(self.bs, np.random.default_rng(seed), Noise(recall=False))
            rescued = _extra_after_subsequence(d_on.units, d_off.units)
            for u in rescued:
                total += 1
                self.assertIsNone(u.hp_frac, seed)          # dropped like a kept unit (unit_hp default ON)
                src = by_cls[u.cls]
                dxdy.append(abs(u.x - src.x) + abs(u.y - src.y))
                unknown_side += int(u.side == -1)
        self.assertGreater(total, 20, "DEGRADE_RECALL=0.855 never dropped a unit worth rescuing over 300 seeds")
        self.assertGreater(float(np.mean(dxdy)), 0.01, "rescued units carry no position noise")
        self.assertAlmostEqual(unknown_side / total, UNKNOWN_TEAM_RATE, delta=0.15)   # loose: small-n binomial

    @patch("pipeline.e1_view.DEGRADE_PRECISION", 1.0)
    def test_hp_none_before_fill_1_0_after_live_view(self):
        for seed in range(60):
            d_on = e1_view._degrade_switchable(self.bs, np.random.default_rng(seed), Noise())
            d_off = e1_view._degrade_switchable(self.bs, np.random.default_rng(seed), Noise(recall=False))
            if not _extra_after_subsequence(d_on.units, d_off.units):
                continue
            v = live_view(self.bs, np.random.default_rng(seed), self.deck, Noise(recall=False))
            self.assertTrue(all(u.hp_frac == 1.0 for u in v.units), seed)   # live fill still applies
            break
        else:
            self.fail("no seed in range(60) rescued a unit -- test is vacuous")

    def test_deterministic_for_a_fixed_seed(self):
        d1 = e1_view._degrade_switchable(self.bs, np.random.default_rng(7), Noise(recall=False))
        d2 = e1_view._degrade_switchable(self.bs, np.random.default_rng(7), Noise(recall=False))
        self.assertEqual(d1, d2)
        v1 = live_view(self.bs, np.random.default_rng(7), self.deck, Noise(recall=False))
        v2 = live_view(self.bs, np.random.default_rng(7), self.deck, Noise(recall=False))
        self.assertEqual(v1, v2)


# ------------------------------------------------------------------------------------------------------
# CLI: --noise-off
# ------------------------------------------------------------------------------------------------------
class TestNoiseOffCli(unittest.TestCase):
    def test_names_match_the_dataclass(self):
        self.assertEqual(set(e1_eval.NOISE_NAMES), set(NOISE_FIELDS))

    def test_default_is_all_on(self):
        n = e1_eval.parse_noise_off("")
        self.assertEqual(n, Noise())
        self.assertEqual(e1_eval.noise_off_names(n), [])

    def test_single_and_multiple_names(self):
        n = e1_eval.parse_noise_off("unit_hp")
        self.assertEqual(n, Noise(unit_hp=False))
        n = e1_eval.parse_noise_off("recall, team , conf")
        self.assertEqual(n, Noise(recall=False, team=False, conf=False))
        self.assertEqual(e1_eval.noise_off_names(n), ["conf", "recall", "team"])

    def test_all_names_off_equals_all_noise_off_constant(self):
        n = e1_eval.parse_noise_off(",".join(NOISE_FIELDS))
        self.assertEqual(n, ALL_NOISE_OFF)

    def test_unknown_name_rejected(self):
        with self.assertRaises(SystemExit):
            e1_eval.parse_noise_off("bogus")
        with self.assertRaises(SystemExit):
            e1_eval.parse_noise_off("recall,not_a_component")

    def test_argparse_wires_the_flag(self):
        a = e1_eval.build_parser().parse_args(["--port", "1", "--out", "x"])
        self.assertEqual(a.noise_off, "")
        a = e1_eval.build_parser().parse_args(["--port", "1", "--out", "x", "--noise-off", "recall,position"])
        self.assertEqual(a.noise_off, "recall,position")
        self.assertEqual(e1_eval.parse_noise_off(a.noise_off), Noise(recall=False, position=False))

    def test_obs_clean_path_never_calls_live_view(self):
        """--obs clean's branch in run_match is `bs` unmodified regardless of --noise-off (module docstring
        of e1_eval / EXPECTED OUTCOME item 5): verified structurally, since clean never reaches live_view."""
        import inspect
        src = inspect.getsource(e1_eval.run_match)
        self.assertIn('live_view(bs, rng_obs, deck, cfg["noise"]) if cfg["obs"] == "live" else bs', src)


if __name__ == "__main__":
    unittest.main()
