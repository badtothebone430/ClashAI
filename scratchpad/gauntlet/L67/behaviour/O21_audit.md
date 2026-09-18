# O21 — audit of the owner's four live-play complaints (LEAD TRANSCRIPTION + lead follow-up)

Scout agents have Read/Glob/Grep only and cannot write; transcribed by the lead 2026-09-18 from the
hand-back, with the scout's own (a)/(b)/(c) labels CORRECTED where it mislabelled "not recoverable"
as "contradicted". Lead's own findings are marked **LEAD**.

## Data used
run20_utf8.log (v6aug tau 0.24, 90 plays / 609 s), run18_utf8.log (164 plays), run12b_utf8.log
(176 plays), session 20260912_152220. **Tonight's live run is NOT recorded** — newest session on
disk is 2026-09-12, so none of this covers the play the owner just watched.

## A. Reaction time (owner: "slowing down")
- (a) Median inter-decision gap **~4.2-4.3 s, stable across run12b / run18 / run20** — no downward
  trend in decision cadence.
- (a) run20 = 8.85 plays/min (90 plays / 609 s).
- (a, prior, HANDOFF 5cs.99 AU/AV) pause share: v6lat tau 0.27 **27%** -> v6aug tau 0.27 **35%** ->
  v6aug tau 0.24 **49%** of match time in >10 s pauses.
- **Reading (b):** what reads as "slower reaction" is almost certainly PAUSE SHARE, not latency —
  the gate no-ops for long stretches and then acts at normal speed. Tau 0.24 (what the owner has
  been running) pauses far more than 0.27. Not yet separated from a possible late-game effect of
  rising unit counts.

## B. Rocket (owner: rare; never finishes a low-HP princess; never rocket+tornado)
- (a) Usage: run20 1/90 = **1.1%**, run12b 2/176 = 1.1%, run18 1/164 = **0.6%**. Complaint CONFIRMED.
- (a) rocket+tornado: run20 tornado 23:33:18.4, rocket 23:36:47.6, tornado 23:36:57.1 — gaps 15.2 s
  and 9.5 s. **Zero pairs within 3 s.** Complaint CONFIRMED.
- **NOT RECOVERABLE (not "contradicted"):** tower HP per play is not in the logs, so "never finishes
  a low-HP princess" can be neither confirmed nor refuted from this data.

## C. Overtime / X-Bow (owner: barely defensive X-Bows; shoves cards at the bridge in OT)
- (a) run20: 4 x_bow / 90 plays = **4.4%**, ALL at board y > 0.5 (own half, i.e. defensive):
  (0.14, 0.61) left and (0.75, 0.61) right. **Zero offensive X-Bows.**
- (b) Bridge concentration is consistent with the placement cluster but NOT quantified against the
  single-elixir mix; **overtime is not marked in the logs**, so OT-only plays cannot be filtered.
  The owner's overtime-specific claim is therefore untested, not disproved.

## D. Tower targeting (owner: never go for the king; 1-0 defend, 1-1 take the other princess)
- (c) **CONTRADICTED — nothing targets the king.** No `king_tower_cell` exists. play.py:343-344
  documents deliberate king avoidance ("a spell that reaches their king wakes it") and masks cells
  accordingly. The owner's hypothesis that bridge-shoving comes from being "forced towards the king
  tower" is not supported by the code.
- (b) Match state (1-0 vs 1-1) is tracked internally by `TowerTracker` but never emitted to the log,
  so play-to-state correlation cannot be done from these logs.

## **LEAD FOLLOW-UP — the doctrine assists ALREADY run on the student path**
`weaker_princess_cell`, `pump_rocket_cell`, `xbow_target_lane_cell` exist in reward.py and are called
from **play.py** (1062, 1065, 1101) and env.py — `student_live.py` calls none of them, which first
looked like "student bypasses doctrine". Reading play.py:1040-1109 shows the opposite:
- :1045-1047 take the STUDENT's chosen `(card_id, cell)` (after the affordability filter).
- :1052+ then operate on that same `cell`: rocket/miner snapped to the **weaker princess**
  (`weaker_princess_cell`), pump-punish rocket snapped king-safe (`pump_rocket_cell`), X-Bow snapped
  to the correct **lane** (`xbow_target_lane_cell`) and then locked (`xbow_lock_cell`), plus an
  X-Bow **pocket** placement when an enemy princess is down (:1084-1091).
So the doctrine layer is post-processing applied to the student's own choice. **The three behaviour
complaints are therefore probably NOT "the model must be retrained to learn this"** — the assist code
exists and runs on the live path; the question is why it does not fire (gating, config, or the
student rarely proposing the card in the first place — rocket is only ~1% of plays, and an assist
that snaps a rocket cannot help if no rocket is ever played).
- Note play.py:1095-1098, a prior recorded observation: `xbow_lock_cell` snaps to the NEARER princess,
  so with a tower already down it "will happily cement a bow into the dead lane -- which is what live
  overtime actually did, several times in one match". That is adjacent to the owner's overtime
  complaint and is already a known defect.

## What this does NOT establish
Why the assists do not fire; whether the owner's overtime observations reproduce (OT unmarked);
tower-HP-conditioned behaviour (not logged); anything about tonight's session (unrecorded).

STATUS: complete (lead transcription + follow-up)
