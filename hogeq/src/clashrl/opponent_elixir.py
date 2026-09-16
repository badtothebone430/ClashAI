"""Opponent elixir inference from observable card-play events.

Live play cannot read the enemy elixir bar directly. This tracker follows the deterministic
relationship

    enemy_elixir ~= my_elixir + my_spend - enemy_spend

where my_elixir is read from OCR, my_spend is known from our own plays, and enemy_spend is inferred
from newly observed enemy card appearances.

The event detector is intentionally simple and robust: enemy detections are matched to short-lived
tracks by (base card, proximity) so one long-lived unit is not charged repeatedly, and clustered on
spawn so swarm cards are charged once per play rather than once per body.
"""
from __future__ import annotations

from typing import Iterable, List


class OpponentElixirEstimator:
    def __init__(self, db, match_radius: float = 0.07, cluster_radius: float = 0.10,
                 forget_s: float = 6.0):
        self.db = db
        self.match_radius = float(match_radius)
        self.cluster_radius = float(cluster_radius)
        self.forget_s = float(forget_s)
        self.reset()

    def reset(self, my_elixir: float = 5.0, now: float | None = None) -> None:
        self._my_spent = 0.0
        self._opp_spent = 0.0
        self._tracks: list[dict] = []      # {base, x, y, t}
        self._est = float(max(0.0, min(10.0, my_elixir)))
        self._rebase = 0.0                 # cumulative saturation correction, for grading the estimator
        self._last_t = float(now) if now is not None else None

    def record_my_play(self, base: str) -> None:
        c = float(self.db.elixir(base) or 0.0)
        if c > 0.0:
            self._my_spent += c

    def _seen_tracks(self, now: float) -> None:
        self._tracks = [tr for tr in self._tracks if now - tr["t"] <= self.forget_s]

    def _find_track(self, base: str, x: float, y: float):
        best = None
        best_d2 = self.match_radius * self.match_radius
        for tr in self._tracks:
            if tr["base"] != base:
                continue
            dx = x - tr["x"]
            dy = y - tr["y"]
            d2 = dx * dx + dy * dy
            if d2 <= best_d2:
                best, best_d2 = tr, d2
        return best

    def _cluster_new(self, fresh: List[tuple[str, float, float]]) -> List[tuple[str, float, float]]:
        """Cluster new detections by base + proximity so swarms cost one card play, not N bodies."""
        out = []
        rem = list(fresh)
        r2 = self.cluster_radius * self.cluster_radius
        while rem:
            base, x, y = rem.pop()
            xs, ys, n = x, y, 1
            keep = []
            for b2, x2, y2 in rem:
                if b2 != base:
                    keep.append((b2, x2, y2))
                    continue
                dx = x2 - x
                dy = y2 - y
                if dx * dx + dy * dy <= r2:
                    xs += x2
                    ys += y2
                    n += 1
                else:
                    keep.append((b2, x2, y2))
            rem = keep
            out.append((base, xs / n, ys / n))
        return out

    def update(self, my_elixir: float, enemy_dets: Iterable, now: float) -> float:
        """Return normalized estimated enemy elixir in [0, 1].

        enemy_dets should contain detections with `.base`, `.cx`, `.gy`, and `.team` where team is
        already resolved to "enemy"/"mine".
        """
        now = float(now)
        self._seen_tracks(now)

        fresh = []
        for d in enemy_dets:
            if getattr(d, "team", None) != "enemy":
                continue
            base = str(getattr(d, "base", "") or "")
            if not base:
                continue
            x = float(getattr(d, "cx", 0.5))
            y = float(getattr(d, "gy", 0.5))
            tr = self._find_track(base, x, y)
            if tr is not None:
                tr["x"], tr["y"], tr["t"] = x, y, now
            else:
                fresh.append((base, x, y))

        for base, x, y in self._cluster_new(fresh):
            self._tracks.append({"base": base, "x": x, "y": y, "t": now})
            c = float(self.db.elixir(base) or 0.0)
            if c > 0.0:
                self._opp_spent += c

        est = float(my_elixir) + self._my_spent - self._opp_spent
        # L67g -- RE-BASELINE ON SATURATION, instead of clipping the output and keeping the bad books.
        # The accounting is exact only if EVERY enemy play is seen: both players regenerate at the same
        # rate, so opp = my_elixir + (what I spent) - (what they spent). Live, the detector misses enemy
        # plays, `_opp_spent` runs low, and `est` drifts UP for the rest of the match -- the clip below used
        # to hide that in the returned value while the next update recomputed from the same inflated base,
        # so the estimate RATCHETED to a pinned 10 and stayed there. Nobody can hold more than 10 elixir, so
        # an overflow IS the evidence of a missed play: charge it to `_opp_spent` and the books recover.
        # Symmetrically, going below 0 means a play was double-counted, so give it back.
        if est > 10.0:
            self._opp_spent += est - 10.0
            self._rebase += est - 10.0                 # diagnostic: total correction charged to the books
            est = 10.0
        elif est < 0.0:
            self._opp_spent += est                     # est < 0 -> reduces _opp_spent
            self._rebase += est
            est = 0.0
        self._est = est
        self._last_t = now
        return self._est / 10.0


# ==========================================================================================================
# TICKET O13 -- OpponentElixirEstimatorV2: a NEW class beside the one above, which stays byte-for-byte
# unchanged (the audit harness's baseline numbers for it must not move). Fixes the over-billing measured in
# O10's charge trace (scratchpad/gauntlet/L67/e1/opp_est/trace_0_5/charges.jsonl, 195 charges over 5 matches
# vs. 70 first_seen): 92 "split" (spawner output sharing/naming the parent's class, or a multi-body card
# scattering past cluster_radius), 25 "rebill_out_of_radius" (fast units outrunning match_radius between
# samples), 6 "rebill_after_expiry". All three ablatable rules default ON; see O13_v2's progress file
# (scratchpad/gauntlet/L67/opp_fix/O13_v2.md) for the data-derived constants and the SPAWNS/BODIES
# verification against pipeline.vocab / CardDB (spawner counts, class names) -- every entry below was cross-
# checked there, not guessed; unverifiable ticket entries (phoenix egg-form, "fire_spirits", "cursed_hog",
# singular "lava_pup") were corrected or dropped, never silently kept wrong.
# ==========================================================================================================

# R1 (suppress_spawns): parent card -> the set of vocab base-keys its spawn output can appear as, PLUS the
# parent's own key (the engine names many spawned bodies after the PARENT card, not the spawned unit --
# pipeline/vocab.py:157-198 documents this; the trace's own `split` reason for tombstone/witch/night_witch is
# entirely SAME-base splits, i.e. exactly this self-naming collision). Verified against
# pipeline.vocab.UNIT_VOCAB and, where available, CardDB.spawner(parent)["unit"] (a measured engine value,
# not a guess) -- see O13_v2.md "SPAWNS / BODIES verification" for the row-by-row source of every entry and
# every correction made to the ticket's own list.
SPAWNS: dict[str, frozenset[str]] = {
    "tombstone": frozenset({"skeletons", "tombstone"}),
    "witch": frozenset({"skeletons", "witch"}),
    "night_witch": frozenset({"bats", "night_witch"}),
    "goblin_hut": frozenset({"spear_goblins", "goblin_hut"}),
    "barbarian_hut": frozenset({"barbarians", "barbarian_hut"}),
    "furnace": frozenset({"fire_spirit", "furnace"}),          # NOT "fire_spirits" -- not a vocab class (O13_v2.md)
    "graveyard": frozenset({"skeletons", "graveyard"}),
    "goblin_drill": frozenset({"goblins", "goblin_drill"}),
    "mother_witch": frozenset({"mother_witch_hog", "mother_witch"}),  # NOT "cursed_hog"/"hog" (not vocab classes)
    "golem": frozenset({"golemite", "golem"}),
    "lava_hound": frozenset({"lava_pups", "lava_hound"}),       # NOT singular "lava_pup" -- not a vocab class
    "elixir_golem": frozenset({"elixir_golemite", "elixir_blob", "elixir_golem"}),
    "skeleton_king": frozenset({"skeletons", "skeleton_king"}),
    # "phoenix" DROPPED -- no egg-form vocab class exists at all (neither "phoenix_egg" nor "egg"); see
    # O13_v2.md UNVERIFIED #1.
}

# R2 (body_count): base -> bodies deployed by one play (or one death-spawn burst). Cross-checked against
# CardDB.get(base)["count"] (icebow/config/cards_stats.json, level-11 wiki import) for every entry, EXCEPT
# lava_pups, sourced instead from CardDB.spawner("lava_hound")["on_death"] == 6 (the card's own `count` field
# describes one pup's stats, not how many spawn on death -- O13_v2.md). "fire_spirits" (ticket: 3) and
# "cursed_hog" (ticket: 1) dropped -- neither is a real vocab class (see SPAWNS comment above); furnace's
# actual (measured) spawn count is 1 fire_spirit at a time, not 3.
BODIES: dict[str, int] = {
    "skeletons": 3, "skeleton_army": 15, "wall_breakers": 2, "barbarians": 5, "elite_barbarians": 2,
    "royal_hogs": 4, "goblin_gang": 6, "goblins": 4, "spear_goblins": 3, "minions": 3, "minion_horde": 6,
    "guards": 3, "three_musketeers": 3, "royal_recruits": 6, "rascals": 3, "bats": 5, "zappies": 3,
    "lava_pups": 6, "goblin_giant": 1, "archers": 2, "dart_goblin": 1, "firecracker": 1, "ice_spirit": 1,
    "fire_spirit": 1, "heal_spirit": 1, "electro_spirit": 1, "mini_pekka": 1,
}

# Every module constant the V2 estimator's default behaviour depends on, logged here so the audit harness can
# dump it verbatim into summary.json (ticket 3) -- see O13_v2.md for the derivation/source of each.
V2_PARAMS: dict[str, object] = {
    "match_radius": 0.07,
    "cluster_radius": 0.10,
    "forget_s": 6.0,
    # R1
    "r_spawn": 0.084,                    # p90 of the trace's 49 tombstone/witch/night_witch split
                                          # prev_same_base_min_dist values (O10_trace / charges.jsonl);
                                          # DATA-DERIVED, not a guess.
    "death_spawn_window_s": 1.0,         # ticket's floor; satisfied for free by forget_s=6.0 (>= 1.0) -- no
                                          # separate timer needed or implemented (O13_v2.md).
    "spawns": SPAWNS,
    # R2
    "body_group_window_s": 1.0,          # W, per ticket ("choose W ~1.0s, state it")
    "r_body": 0.20,                      # NOT data-derived -- a judgment call (O13_v2.md), ~2x cluster_radius
    "bodies": BODIES,
    # R3
    "speed_k": 1.5,
    "speed_radius_cap": 0.25,
    "speed_axis": "x:/18 (larger of x:/18, y:/32 per-axis normalized speed)",
}


class OpponentElixirEstimatorV2:
    """Same public API as OpponentElixirEstimator (reset/record_my_play/update/._est), same exact-books
    saturation re-baselining (copied verbatim from L67g below), plus three independently-ablatable fixes for
    the over-billing O10's trace measured. Each defaults ON; set any to False to reproduce
    OpponentElixirEstimator's exact behaviour for that rule (with all three False, this class is behaviourally
    identical to OpponentElixirEstimator -- same tracks, same matches, same one-charge-per-cluster, verified
    by TestAblationMatchesV1 in the new test file).

    R1 suppress_spawns: a new same/related-base track within `r_spawn` of an existing LIVE track of a base in
        SPAWNS is created WITHOUT a charge (it still becomes a track, so it stops re-triggering next frame).
    R2 body_count: new same-base tracks from one update() call are grouped and charged
        ceil(n_new / BODIES[base]) times (not once per cluster); further same-base arrivals within
        `body_group_window_s` seconds and `r_body` of the group's centroid are absorbed free, up to
        BODIES[base] bodies total.
    R3 speed_aware_radius: match radius per det is widened by the card's own `db.speed_tiles(base)`
        (icebow/config/cards_stats.json), `max(match_radius, speed_norm*dt*speed_k)` capped at
        `speed_radius_cap` -- see O13_v2.md for a measured gap (bandit) this does not close.
    """

    def __init__(self, db, match_radius: float = 0.07, cluster_radius: float = 0.10,
                 forget_s: float = 6.0, suppress_spawns: bool = True, body_count: bool = True,
                 speed_aware_radius: bool = True, spawns: dict | None = None, bodies: dict | None = None,
                 r_spawn: float = V2_PARAMS["r_spawn"], r_body: float = V2_PARAMS["r_body"],
                 body_group_window_s: float = V2_PARAMS["body_group_window_s"],
                 speed_k: float = V2_PARAMS["speed_k"], speed_radius_cap: float = V2_PARAMS["speed_radius_cap"]):
        self.db = db
        self.match_radius = float(match_radius)
        self.cluster_radius = float(cluster_radius)
        self.forget_s = float(forget_s)
        self.suppress_spawns = bool(suppress_spawns)
        self.body_count = bool(body_count)
        self.speed_aware_radius = bool(speed_aware_radius)
        self.spawns = dict(spawns) if spawns is not None else SPAWNS
        self.bodies = dict(bodies) if bodies is not None else BODIES
        self.r_spawn = float(r_spawn)
        self.r_body = float(r_body)
        self.body_group_window_s = float(body_group_window_s)
        self.speed_k = float(speed_k)
        self.speed_radius_cap = float(speed_radius_cap)
        self.reset()

    def reset(self, my_elixir: float = 5.0, now: float | None = None) -> None:
        self._my_spent = 0.0
        self._opp_spent = 0.0
        self._tracks: list[dict] = []      # {base, x, y, t}
        self._body_groups: dict[str, dict] = {}   # base -> {cx, cy, n, t_start}  (R2 pending deployments)
        self._est = float(max(0.0, min(10.0, my_elixir)))
        self._rebase = 0.0
        self._last_t = float(now) if now is not None else None
        # O13 attempt 2 FIX 3: per-NEW-TRACK (base, x, y, charged, cost) ledger for this update() call, in
        # creation order -- see the comment in update() for why this exists (unambiguous charge attribution
        # for a downstream trace, kept OUT of V1 which has no such ambiguity).
        self.last_new_tracks: list[tuple[str, float, float, bool, float]] = []

    def record_my_play(self, base: str) -> None:
        c = float(self.db.elixir(base) or 0.0)
        if c > 0.0:
            self._my_spent += c

    def _seen_tracks(self, now: float) -> None:
        # forget_s (unchanged, 6.0 by default) keeps a dead spawner's track live well past R1's requested
        # >=1.0s "still counts as live" floor -- see O13_v2.md; no separate death-spawn timer is needed.
        self._tracks = [tr for tr in self._tracks if now - tr["t"] <= self.forget_s]

    def _speed_radius(self, base: str, dt: float) -> float:
        if not self.speed_aware_radius:
            return self.match_radius
        speed_tiles = None
        get_speed = getattr(self.db, "speed_tiles", None)
        if callable(get_speed):
            speed_tiles = get_speed(base)
        if speed_tiles is None or dt <= 0.0:
            return self.match_radius
        speed_norm = float(speed_tiles) / 18.0     # the larger of x:/18, y:/32 (O13_v2.md)
        eff = speed_norm * float(dt) * self.speed_k
        eff = min(eff, self.speed_radius_cap)
        return max(self.match_radius, eff)

    def _find_track(self, base: str, x: float, y: float, now: float):
        best = None
        best_d2 = None
        for tr in self._tracks:
            if tr["base"] != base:
                continue
            r = self._speed_radius(base, now - tr["t"])
            dx = x - tr["x"]
            dy = y - tr["y"]
            d2 = dx * dx + dy * dy
            if d2 <= r * r and (best_d2 is None or d2 <= best_d2):
                best, best_d2 = tr, d2
        return best

    def _cluster_new(self, fresh: List[tuple[str, float, float]]) -> List[tuple[str, float, float, int]]:
        """Same clustering as V1 (base + cluster_radius proximity), but also returns each cluster's member
        count `n` -- R2 needs body counts, not just centroids."""
        out = []
        rem = list(fresh)
        r2 = self.cluster_radius * self.cluster_radius
        while rem:
            base, x, y = rem.pop()
            xs, ys, n = x, y, 1
            keep = []
            for b2, x2, y2 in rem:
                if b2 != base:
                    keep.append((b2, x2, y2))
                    continue
                dx = x2 - x
                dy = y2 - y
                if dx * dx + dy * dy <= r2:
                    xs += x2
                    ys += y2
                    n += 1
                else:
                    keep.append((b2, x2, y2))
            rem = keep
            out.append((base, xs / n, ys / n, n))
        return out

    def _spawn_suppressed(self, base: str, x: float, y: float) -> bool:
        """R1: is (base, x, y) within r_spawn of a LIVE track whose base P has `base` in SPAWNS[P]
        (P's own key included, per the self-naming collision -- see SPAWNS comment)?"""
        if not self.suppress_spawns:
            return False
        r2 = self.r_spawn * self.r_spawn
        for tr in self._tracks:
            spawn_set = self.spawns.get(tr["base"])
            if not spawn_set or base not in spawn_set:
                continue
            dx = x - tr["x"]
            dy = y - tr["y"]
            if dx * dx + dy * dy <= r2:
                return True
        return False

    def _charge_groups(self, base: str, points: list[tuple[float, float, int]], now: float) -> int:
        """R2: `points` are this update's non-suppressed (base, x, y, n) cluster points for one base.
        Returns the number of `db.elixir(base)` calls to charge (0 when fully absorbed into an active
        group). Always advances/creates `self._body_groups[base]` bookkeeping regardless of `body_count`,
        so turning the rule on mid-match never sees stale state -- but only CHARGES per-cluster (V1 parity)
        when `body_count` is False."""
        cap = self.bodies.get(base)
        if not self.body_count or not cap or cap <= 1:
            return len(points)          # V1 parity: one charge per cluster point
        total_n = sum(p[2] for p in points)
        if total_n <= 0:
            return 0
        wx = sum(p[0] * p[2] for p in points) / total_n
        wy = sum(p[1] * p[2] for p in points) / total_n
        grp = self._body_groups.get(base)
        if (grp is not None and (now - grp["t_start"]) <= self.body_group_window_s
                and grp["n"] < grp["cap"]):
            dx, dy = wx - grp["cx"], wy - grp["cy"]
            if dx * dx + dy * dy <= self.r_body * self.r_body:
                grp["n"] = min(grp["n"] + total_n, grp["cap"])
                return 0                # absorbed into the active deployment, no new charge
        # start a new deployment
        charges = -(-total_n // cap)    # ceil
        self._body_groups[base] = {"cx": wx, "cy": wy, "n": min(total_n, charges * cap),
                                   "cap": charges * cap, "t_start": now}
        return charges

    def update(self, my_elixir: float, enemy_dets: Iterable, now: float) -> float:
        """Return normalized estimated enemy elixir in [0, 1]. Same det shape as V1
        (`.base`, `.cx`, `.gy`, `.team`)."""
        now = float(now)
        self._seen_tracks(now)

        fresh = []
        for d in enemy_dets:
            if getattr(d, "team", None) != "enemy":
                continue
            base = str(getattr(d, "base", "") or "")
            if not base:
                continue
            x = float(getattr(d, "cx", 0.5))
            y = float(getattr(d, "gy", 0.5))
            tr = self._find_track(base, x, y, now)
            if tr is not None:
                tr["x"], tr["y"], tr["t"] = x, y, now
            else:
                fresh.append((base, x, y))

        clusters = self._cluster_new(fresh)          # [(base, x, y, n)]
        by_base: dict[str, list[tuple[float, float, int]]] = {}
        suppressed: list[bool] = []
        for base, x, y, n in clusters:
            supp = self._spawn_suppressed(base, x, y)
            suppressed.append(supp)
            if not supp:
                by_base.setdefault(base, []).append((x, y, n))

        n_charges_for_base: dict[str, int] = {
            base: self._charge_groups(base, pts, now) for base, pts in by_base.items()
        }
        remaining: dict[str, int] = dict(n_charges_for_base)

        # O13 attempt 2 FIX 3 (verifier, MEDIUM): record, per NEW TRACK created this call, whether it was
        # actually charged and for how much -- in creation order, 1:1 with the tracks appended below. A
        # positional zip of "new tracks this update" against "db.elixir() calls this update" (which is how
        # the harness's TracedEstimator attributes a charge to a track) is only valid when EVERY new track
        # makes EXACTLY one call, which is true for V1 but false for V2: a spawn-suppressed (R1) or body-
        # count-absorbed (R2) track makes NO call at all, so a no-charge track ahead of a charged one in the
        # SAME update would shift every later (base, x, y) attribution off by one. Exposing this explicit,
        # unambiguous per-track ledger (kept OUT of V1, which has no such ambiguity) lets the harness
        # attribute correctly with no re-implementation of R1/R2's own logic.
        self.last_new_tracks: list[tuple[str, float, float, bool, float]] = []
        for (base, x, y, n), supp in zip(clusters, suppressed):
            self._tracks.append({"base": base, "x": x, "y": y, "t": now})
            charged, cost = False, 0.0
            if not supp and remaining.get(base, 0) > 0:
                remaining[base] -= 1
                c = float(self.db.elixir(base) or 0.0)
                if c > 0.0:
                    self._opp_spent += c
                    charged, cost = True, c
            self.last_new_tracks.append((base, x, y, charged, cost))

        est = float(my_elixir) + self._my_spent - self._opp_spent
        # L67g -- RE-BASELINE ON SATURATION (copied verbatim from OpponentElixirEstimator.update above).
        if est > 10.0:
            self._opp_spent += est - 10.0
            self._rebase += est - 10.0
            est = 10.0
        elif est < 0.0:
            self._opp_spent += est
            self._rebase += est
            est = 0.0
        self._est = est
        self._last_t = now
        return self._est / 10.0