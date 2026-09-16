# O5 -- split `scalars` into `my_elixir` / `opp_elixir` / `king_hp`

Repo HEAD at start: `4bca15920e6cc5f0046b3952af28ea924326b1fe` (matches ticket). Write set touched:
`pipeline/e1_view.py`, `pipeline/e1_eval.py`, `pipeline/tests/test_e1_noise_arms.py`. `pipeline/obs_contract.py`
NOT touched (verified, see acceptance g below). No engine run, no commit.

## What changed

`e1_view.Noise` field order is now:
`recall, false_pos, position, team, unit_hp, my_elixir, opp_elixir, king_hp, deploying, conf` (10 fields,
`scalars` removed as a dataclass field). Consumption sites:

- `_degrade_switchable` line ~163 (was 158-159): king tower `hp_frac -> None` now gated on `noise.king_hp`
  (was `noise.scalars`).
- same function, ~166-168 (was 161-163): `my_elixir` floor + `my_elixir_exact` clear now gated on
  `noise.my_elixir`; `opp_elixir -> None` now gated on `noise.opp_elixir`.
- `live_view` line ~188 (was 183-184): king `hp_frac -> KING_HP_LIVE` fill now gated on `noise.king_hp`.

`ALL_NOISE_OFF` updated to set all three new fields False (replacing the single `scalars=False`).
Field docstrings replaced the one `scalars` bullet with three bullets (`my_elixir`, `opp_elixir`, `king_hp`),
each noting "(O5 split of the old `scalars` switch)".

`e1_eval.py`:
- `NOISE_NAMES` comment updated ("8" -> "10" component names) -- the tuple itself is still derived from the
  dataclass fields, so it auto-updated to the 10 real names.
- New `NOISE_ALIASES = {"scalars": ("my_elixir", "opp_elixir", "king_hp")}`.
- `parse_noise_off` now expands any name through `NOISE_ALIASES` before validating against `NOISE_NAMES`, so
  `--noise-off scalars` (and combinations like `scalars,recall`) still parse, and old `run.json` files that
  recorded `"noise_off": ["scalars"]` remain a valid re-parse of the same CLI spelling. `noise_off_names`
  itself is unchanged -- it always returns real field names (confirmed: `noise_off_names(parse_noise_off(
  "scalars")) == ["king_hp", "my_elixir", "opp_elixir"]`).
- `--noise-off` help string lists the 10 real names plus a note on the `scalars` alias and what it expands to.

Found via `grep -r scalars scratchpad/gauntlet/L67/e1`: several existing run dirs
(`attrib_noise/off_scalars*/slot0/run.json`) do record `"noise_off": ["scalars"]` from past runs -- this is
exactly why the ticket asked for the alias; confirmed the need is real, not hypothetical. Those files are
untouched (out of write set).

## Test changes (`pipeline/tests/test_e1_noise_arms.py`)

- `NOISE_FIELDS` and the `field_only` tuple in `TestRngStability` updated to the 3 new names in place of
  `scalars`.
- `test_field_components_reduce_to_true_engine_values`: the one `Noise(scalars=False)` block replaced with
  three separate checks (`my_elixir=False`, `opp_elixir=False`, `king_hp=False`), the last against king tower
  `hp_frac` (this fixture's kings are undamaged, hp_frac 1.0, so this particular assertion is not a strong
  discriminator by itself -- see the new isolation tests below for a damaged-king fixture that is).
- **New (acceptance b): golden test.** `_O5_GOLDEN_SCALARS_OFF`, a `dataclasses.asdict()` dump of
  `live_view(bs, rng, deck, Noise(scalars=False))` on 3 boards/seeds (`_bs(deck)`/seed 0,
  `_bs(deck, extra_enemy=0)`/seed 1, `raw_obs_board(deck)`/seed 2 -- the exact fixtures/pattern
  `TestAllOnBitIdentical` already uses). **Captured BEFORE any edit**, by writing a standalone script
  (`capture_o5_golden.py`, contents below) that imports the then-unedited `pipeline.e1_view` (single
  `scalars` field still present) and prints the `pprint.pformat` of the 3 asdict dumps; that printed output
  was pasted verbatim into the test file as `_O5_GOLDEN_SCALARS_OFF`. `TestGoldenSplitMatchesPreO5Scalars`
  then asserts `dataclasses.asdict(live_view(bs, rng, deck, Noise(my_elixir=False, opp_elixir=False,
  king_hp=False))) == golden` for each of the 3 (board, seed) pairs, using the POST-edit code -- passes.
- **New (acceptance c): `TestThreeSwitchesIsolated`.** Fixture: `_bs(deck)` (my_elixir=6.37 fractional,
  opp_elixir=3.0 -- both already non-trivial) with both king towers damaged via `replace()`
  (hp_frac 0.42 / 0.77, since the stock fixture's kings sit at 1.0 and wouldn't discriminate the switch).
  Three tests (`my_elixir`, `opp_elixir`, `king_hp` each off alone) each assert: its own target field(s)
  change to the true/unfloored/kept value, the OTHER two scalar targets are unchanged, `v0.units == v1.units`
  and `v0.spells == v1.spells` (byte-identical -- these switches never touch the RNG-consuming `pass_()`
  path), and for `king_hp` specifically that the non-king (princess) towers are untouched and the king
  towers show a genuinely different value (not just "still equal because untested").
- **New (acceptance d): CLI alias tests** in `TestNoiseOffCli`:
  `test_scalars_alias_expands_to_the_three_split_fields`,
  `test_scalars_alias_combines_with_a_plain_name` (`"scalars,recall"` also clears `recall`),
  `test_scalars_alias_still_rejects_an_unknown_name_alongside_it` (`"scalars,bogus"` -> SystemExit).
  The pre-existing `test_unknown_name_rejected` (plain "bogus") is untouched and still passes.

## Golden capture script (run once, BEFORE editing, against HEAD 4bca159 unedited code)

```python
import dataclasses, pprint, sys
from pathlib import Path
REPO = Path(r"C:\Users\benpe\ClashBot")
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
import numpy as np
from pipeline import obs_contract as oc
from pipeline.e1_view import Noise, live_view
from pipeline.tests.test_obs_contract import ENGINE_DECK, raw_obs

def _deck(): return oc.load_deck("icebow")

def _bs(deck, *, extra_enemy=6):
    from pipeline import engine_play as ep
    o = raw_obs(0)
    names = ["HogRider","Musketeer","Valkyrie","Giant","MiniPekka","Wizard"]
    for i in range(extra_enemy):
        o["entities"].append({"side":1,"x":3000+2000*i,"y":18000+500*i,"card_id":26000100+i,
                              "name":names[i % len(names)],"hp":500,"max_hp":1000,"kind":15})
    return oc.from_engine(ep.compact_raw(o), 0, deck, engine_deck=ENGINE_DECK, unmapped=set())

def raw_obs_board(deck):
    from pipeline import engine_play as ep
    return oc.from_engine(ep.compact_raw(raw_obs(1)), 1, deck, engine_deck=ENGINE_DECK, unmapped=set())

deck = _deck()
boards_seeds = [(_bs(deck), 0), (_bs(deck, extra_enemy=0), 1), (raw_obs_board(deck), 2)]
golden = [dataclasses.asdict(live_view(bs, np.random.default_rng(seed), deck, Noise(scalars=False)))
          for bs, seed in boards_seeds]
print("_O5_GOLDEN_SCALARS_OFF = ", end=""); print(pprint.pformat(golden, width=110))
```
Ran with `icebow/.venv/Scripts/python.exe`, output redirected to a scratch file, then pasted verbatim into
the test file (this is the literal output -- no hand-editing of values).

## Verification (all evidence, this session)

a. **`python -m pytest pipeline/tests/test_e1_noise_arms.py -q`** (pytest was not installed in
   `icebow/.venv`; installed it via `pip install pytest` into that venv first -- a test-runner-only addition,
   no pipeline code dependency added):
   ```
   ..............................                                           [100%]
   30 passed in 2.24s
   ```
   (23 original + 7 new: 1 golden + 3 isolation + 3 alias.) Also cross-checked with
   `python -m unittest pipeline.tests.test_e1_noise_arms -v` (the invocation the test module's own docstring
   names) -- also 30/30 OK, same tests.

b. Golden capture method: described above; `TestGoldenSplitMatchesPreO5Scalars.
   test_three_switches_off_matches_scalars_off_golden` passes post-edit (see (a)).

c. `TestThreeSwitchesIsolated` (3 tests) passes -- see (a). Confirmed by code inspection too: `my_elixir`,
   `opp_elixir`, `king_hp` are read only in the two `replace(bs/d, ...)` calls that build `BoardState`
   scalars/towers; `pass_()` (the only RNG-consuming code) never reads them, so `units`/`spells` cannot
   diverge for any seed.

d. `TestNoiseOffCli` alias tests (3 new + 1 pre-existing unknown-name test) pass -- see (a).

e. `TestAllOnBitIdentical` (both tests, pre-existing, unchanged in meaning) pass -- see (a). `Noise()` (all
   True, all 10 fields) still bit-identical to `obs_contract.degrade` + live fill.

f. `python -m pipeline.e1_eval --help` exit 0; `--noise-off` help text:
   `recall,false_pos,position,team,unit_hp,my_elixir,opp_elixir,king_hp,deploying,conf (plus alias 'scalars'
   = my_elixir,opp_elixir,king_hp)`.

g. `git diff --stat pipeline/obs_contract.py` -> empty (no output). Confirmed untouched.

`git diff --stat` for the write set:
```
 pipeline/e1_eval.py                  |  18 +-
 pipeline/e1_view.py                  |  23 ++-
 pipeline/tests/test_e1_noise_arms.py | 352 ++++++++++++++++++++++++++++++++++-
 3 files changed, 372 insertions(+), 21 deletions(-)
```
`git status --short pipeline/` shows only these 3 tracked files modified; nothing else in `pipeline/` touched.

## Concerns / things I'm not fully sure of

- Installed `pytest` into `icebow/.venv` (it was absent) to satisfy the ticket's literal verify command.
  This changes venv package state (not a pipeline code dependency, and the ticket's own "no new dependencies"
  is under CONSTRAINTS for the code change, not the verify step) -- flagging in case that's unwanted; the
  same 30/30 result was independently confirmed via `python -m unittest`, so pytest's presence isn't load-
  bearing for the actual acceptance evidence.
- `test_field_components_reduce_to_true_engine_values`'s `king_hp=False` check (in `TestRngStability`, the
  pre-existing test I minimally updated) checks against a fixture where both kings are already at hp_frac
  1.0, so it doesn't by itself prove `king_hp` changes anything -- I left it as a direct port of the old
  `scalars` assertion pattern and added the real discriminating test (`TestThreeSwitchesIsolated`, damaged
  kings) as the meaningful acceptance-c evidence instead of trying to retrofit that older test.
- Did not run the engine, did not touch HANDOFF.md, did not commit, per ticket scope.

STATUS: complete
