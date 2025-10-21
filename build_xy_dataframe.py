#!/usr/bin/env python3
"""
Build a per-frame LoL timeline dataset with X (inputs) and Y (targets).

- Consumes timeline JSONs saved by RiotTimelineParser (default: timeline_data/).
- Produces a flat DataFrame with:
  timestamp, match_id, frame_idx, puuid, team,
  X_* feature columns (ChampionStats, DamageStats, gold, CS, pos, event counts),
  Y_* target columns (current_gold, total_gold, total_xp, total_kills).

Usage:
    python build_xy_dataframe.py --timeline_dir timeline_data --out xy_rows.parquet
    # or CSV
    python build_xy_dataframe.py --timeline_dir timeline_data --out xy_rows.csv

If you want to compute the 10-dim X vector via your hidden f, implement apply_f(...)
and run with --compute_f to append columns Xf_p1 ... Xf_p10 per row.
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

# ----------------------------
# Utilities
# ----------------------------

def _safe_get(d: Dict, key: str, default=None):
    return d.get(key, default) if isinstance(d, dict) else default

def _flatten_prefixed(prefix: str, d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a 1-level dict of scalars under a prefix.
    Non-numeric values are kept if trivially serializable; missing -> ignored.
    """
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out[f"{prefix}{k}"] = v
    return out

def _participant_id_to_team(pid: int) -> int:
    # Standard mapping: 1..5 blue (team 100), 6..10 red (team 200)
    return 1 if 1 <= pid <= 5 else -1

def _map_participant_id_to_puuid(tl: Dict[str, Any]) -> Dict[int, str]:
    """
    Return {participantId -> puuid}. Handles two common layouts:
      - timeline['metadata']['participants'] == [puuid1,...,puuid10] in pid order
      - timeline['info']['participants'] as list of dicts with 'participantId' and 'puuid'
        (or rarely, list of puuids)
    """
    mapping: Dict[int, str] = {}
    meta = _safe_get(tl, "metadata", {})
    info = _safe_get(tl, "info", {})

    # Preferred: metadata.participants list (ordered by pid)
    meta_participants = _safe_get(meta, "participants", None)
    if isinstance(meta_participants, list) and len(meta_participants) == 10:
        for i, puuid in enumerate(meta_participants, start=1):
            mapping[i] = puuid
        return mapping

    # Fallback: info.participants may be list[dict] or list[str]
    info_participants = _safe_get(info, "participants", None)
    if isinstance(info_participants, list) and len(info_participants) == 10:
        if all(isinstance(x, dict) for x in info_participants):
            for p in info_participants:
                pid = int(_safe_get(p, "participantId", 0))
                puuid = _safe_get(p, "puuid", "")
                if pid and puuid:
                    mapping[pid] = puuid
        else:
            # assume ordered list of puuids
            for i, puuid in enumerate(info_participants, start=1):
                mapping[i] = puuid
    return mapping

def _summarize_events_for_frame(events: List[Dict[str, Any]]) -> Dict[int, Dict[str, int]]:
    """
    Build per-participant counts for key event types within a single frame.
    Returns: {pid: {'kills':..., 'deaths':..., 'assists':..., 'elite':..., 'buildings':..., 'ward_placed':..., 'ward_killed':...}}
    """
    per_pid = defaultdict(lambda: defaultdict(int))

    if not isinstance(events, list):
        return per_pid

    for ev in events:
        et = _safe_get(ev, "type", "")
        if et == "CHAMPION_KILL":
            killer = int(_safe_get(ev, "killerId", 0) or 0)
            victim = int(_safe_get(ev, "victimId", 0) or 0)
            assists = _safe_get(ev, "assistingParticipantIds", []) or []
            if killer > 0:
                per_pid[killer]["kills"] += 1
            if victim > 0:
                per_pid[victim]["deaths"] += 1
            for a in assists:
                if a > 0:
                    per_pid[a]["assists"] += 1

        elif et == "ELITE_MONSTER_KILL":
            killer = int(_safe_get(ev, "killerId", 0) or 0)
            if killer > 0:
                per_pid[killer]["elite"] += 1

        elif et == "BUILDING_KILL":
            killer = int(_safe_get(ev, "killerId", 0) or 0)
            if killer > 0:
                per_pid[killer]["buildings"] += 1

        elif et == "WARD_PLACED":
            creator = int(_safe_get(ev, "creatorId", 0) or 0)
            if creator > 0:
                per_pid[creator]["ward_placed"] += 1

        elif et == "WARD_KILL":
            killer = int(_safe_get(ev, "killerId", 0) or 0)
            if killer > 0:
                per_pid[killer]["ward_killed"] += 1

    return per_pid

# ----------------------------
# Optional hook for your hidden f
# ----------------------------

def apply_f(all_10_players_feature_vector: np.ndarray) -> float:
    """
    Placeholder for your hidden function f: R^n -> R (n=10).
    Given a 10-dim vector for *this* timestamp (e.g., one scalar per player),
    return a single real value.

    Currently returns None (stub). Replace this with your actual logic if/when available.
    """
    return None

# ----------------------------
# Core extraction
# ----------------------------

def rows_from_timeline(tl: Dict[str, Any], match_id: str,
                       compute_f: bool = False) -> List[Dict[str, Any]]:
    """
    Convert a single timeline JSON to row dicts for a DataFrame.
    """
    info = _safe_get(tl, "info", {})
    frames = _safe_get(info, "frames", []) or []
    pid_to_puuid = _map_participant_id_to_puuid(tl)

    # cumulative kills per participant (to build Y_total_kills)
    cum_kills = defaultdict(int)

    rows: List[Dict[str, Any]] = []
    for f_idx, fr in enumerate(frames):
        ts = int(_safe_get(fr, "timestamp", 0) or 0)
        pframes = _safe_get(fr, "participantFrames", {}) or {}
        events = _safe_get(fr, "events", []) or []

        # summarize events for this frame, then update cumulative kills
        evt_counts = _summarize_events_for_frame(events)
        for pid, d in evt_counts.items():
            if "kills" in d:
                cum_kills[pid] += d["kills"]

        # Optionally: prepare a 10-dim vector for f (your choice of scalar per player)
        # Here as an example we use each player's totalGold at this frame.
        f_vector = np.full(10, np.nan)
        for pid_int in range(1, 11):
            pf = _safe_get(pframes, str(pid_int), {}) or {}
            f_vector[pid_int - 1] = _safe_get(pf, "totalGold", np.nan)

        f_value = apply_f(f_vector) if compute_f else None

        # Now create a row per participant for this frame
        for pid_str, pf in pframes.items():
            try:
                pid = int(pid_str)
            except Exception:
                continue

            puuid = pid_to_puuid.get(pid, "")
            team = _participant_id_to_team(pid)

            # ChampionStats (dict) and DamageStats (dict) flatten
            champ_stats = _safe_get(pf, "championStats", {}) or _safe_get(pf, "champIsoStats", {}) or {}
            dmg_stats = _safe_get(pf, "damageStats", {}) or {}

            champ_flat = _flatten_prefixed("X_champ_", champ_stats)
            dmg_flat = _flatten_prefixed("X_dmg_", dmg_stats)

            pos = _safe_get(pf, "position", {}) or {}
            pos_x = _safe_get(pos, "x", np.nan)
            pos_y = _safe_get(pos, "y", np.nan)

            # Per-frame event counts for this participant
            ec = evt_counts.get(pid, {})
            evt_feats = {
                "X_evt_kills": int(ec.get("kills", 0)),
                "X_evt_deaths": int(ec.get("deaths", 0)),
                "X_evt_assists": int(ec.get("assists", 0)),
                "X_evt_elite_monsters": int(ec.get("elite", 0)),
                "X_evt_buildings": int(ec.get("buildings", 0)),
                "X_evt_wards_placed": int(ec.get("ward_placed", 0)),
                "X_evt_wards_killed": int(ec.get("ward_killed", 0)),
            }

            row = {
                # keys
                "match_id": match_id,
                "frame_idx": f_idx,
                "timestamp": ts,
                "participantId": pid,
                "puuid": puuid,
                "team": team,

                # X (inputs to f)
                "X_team": team,  # explicit in case you train on X_* only
                "X_current_gold": _safe_get(pf, "currentGold", np.nan),
                "X_minions_killed": _safe_get(pf, "minionsKilled", np.nan),
                "X_jungle_minions_killed": _safe_get(pf, "jungleMinionsKilled", np.nan),
                "X_time_enemy_spent_controlled": _safe_get(pf, "timeEnemySpentControlled", np.nan),
                "X_pos_x": pos_x,
                "X_pos_y": pos_y,

                # Y (targets)
                "Y_current_gold": _safe_get(pf, "currentGold", np.nan),
                "Y_total_gold": _safe_get(pf, "totalGold", np.nan),
                "Y_total_xp": _safe_get(pf, "xp", np.nan),
                "Y_total_kills": int(cum_kills.get(pid, 0)),
            }

            # Optional: include a per-frame computed f value shared for all rows in this frame
            # (Enable via --compute_f; otherwise these are omitted below when building DataFrame)
            if compute_f:
                row["Xf_this_frame"] = f_value

            # Merge flattened dicts
            row.update(champ_flat)
            row.update(dmg_flat)
            row.update(evt_feats)

            rows.append(row)

    return rows

def build_xy_dataframe(timeline_dir: str, compute_f: bool = False) -> pd.DataFrame:
    """
    Walk a directory of *_timeline.json files and build the unified XY DataFrame.
    """
    paths = sorted(glob.glob(os.path.join(timeline_dir, "*_timeline.json")))
    all_rows: List[Dict[str, Any]] = []

    for p in paths:
        match_id = os.path.basename(p).replace("_timeline.json", "")
        try:
            with open(p, "r", encoding="utf-8") as f:
                tl = json.load(f)
            rows = rows_from_timeline(tl, match_id, compute_f=compute_f)
            all_rows.extend(rows)
        except Exception as e:
            print(f"[WARN] Failed to parse {p}: {e}")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Ensure core columns exist even if some timelines lacked data
    core_cols = [
        "timestamp", "match_id", "frame_idx", "puuid", "team",
        "X_team", "X_current_gold", "X_minions_killed", "X_jungle_minions_killed",
        "X_time_enemy_spent_controlled", "X_pos_x", "X_pos_y",
        "Y_current_gold", "Y_total_gold", "Y_total_xp", "Y_total_kills",
    ]
    for c in core_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Sort for convenience
    df = df.sort_values(["match_id", "frame_idx", "participantId"]).reset_index(drop=True)

    # Drop duplicate rows if any
    df = df.drop_duplicates(subset=["match_id", "frame_idx", "participantId"], keep="last")

    # If compute_f=False, drop the placeholder column if present
    if "Xf_this_frame" in df.columns and not compute_f:
        df = df.drop(columns=["Xf_this_frame"])

    return df

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Build LoL XY dataframe from timeline JSONs.")
    ap.add_argument("--timeline_dir", type=str, default="timeline_data", help="Directory with *_timeline.json files.")
    ap.add_argument("--out", type=str, required=True, help="Output path (.csv or .parquet).")
    ap.add_argument("--compute_f", action="store_true", help="Compute Xf_this_frame via apply_f (stub).")
    args = ap.parse_args()

    df = build_xy_dataframe(args.timeline_dir, compute_f=args.compute_f)

    if df.empty:
        print(f"No timeline files found in {args.timeline_dir} or no rows parsed.")
        return

    # Save
    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if out.lower().endswith(".csv"):
        df.to_csv(out, index=False)
    elif out.lower().endswith(".parquet"):
        df.to_parquet(out, index=False)
    else:
        # default to CSV
        if "." not in os.path.basename(out):
            out += ".csv"
        df.to_csv(out, index=False)

    print(f"Saved {len(df):,} rows -> {out}")

if __name__ == "__main__":
    main()
