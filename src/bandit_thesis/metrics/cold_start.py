from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _user_id(row: Dict[str, Any]) -> int:
    if "user_id" in row:
        return int(row["user_id"])
    return int(row["context"]["user_id"])


def cold_start_ctr_per_user(rows: List[Dict[str, Any]], m: int = 20) -> float:
    user_counts: Dict[int, int] = {}
    user_clicks_first_m: Dict[int, int] = {}

    for row in rows:
        user_id = _user_id(row)
        count = user_counts.get(user_id, 0)
        if count < m:
            user_clicks_first_m[user_id] = user_clicks_first_m.get(user_id, 0) + int(row["reward"])
        user_counts[user_id] = count + 1

    eligible = [user_id for user_id, count in user_counts.items() if count >= m]
    if not eligible:
        return float("nan")

    values = [user_clicks_first_m.get(user_id, 0) / m for user_id in eligible]
    return float(np.nanmean(values))


def cold_start_regret_per_user(rows: List[Dict[str, Any]], m: int = 20) -> float:
    user_counts: Dict[int, int] = {}
    user_regret_first_m: Dict[int, float] = {}

    for row in rows:
        user_id = _user_id(row)
        count = user_counts.get(user_id, 0)
        if count < m:
            regret = float(row["p_opt"]) - float(row["p_chosen"])
            user_regret_first_m[user_id] = user_regret_first_m.get(user_id, 0.0) + regret
        user_counts[user_id] = count + 1

    eligible = [user_id for user_id, count in user_counts.items() if count >= m]
    if not eligible:
        return float("nan")

    values = [user_regret_first_m.get(user_id, 0.0) for user_id in eligible]
    return float(np.nanmean(values))


def ctr_at_n(rows: List[Dict[str, Any]], n: int = 1000) -> float:
    n = min(n, len(rows))
    if n <= 0:
        return float("nan")
    clicks = sum(int(row["reward"]) for row in rows[:n])
    return float(clicks / n)


def regret_at_n(rows: List[Dict[str, Any]], n: int = 1000) -> float:
    n = min(n, len(rows))
    if n <= 0:
        return float("nan")
    return float(sum(float(row["p_opt"]) - float(row["p_chosen"]) for row in rows[:n]))


def ctr_last_w(rows: List[Dict[str, Any]], w: int = 500) -> float:
    if not rows:
        return float("nan")
    tail = rows[-min(w, len(rows)) :]
    clicks = sum(int(row["reward"]) for row in tail)
    return float(clicks / len(tail))


def regret_last_w(rows: List[Dict[str, Any]], w: int = 500) -> float:
    if not rows:
        return float("nan")
    tail = rows[-min(w, len(rows)) :]
    return float(sum(float(row["p_opt"]) - float(row["p_chosen"]) for row in tail))


def ctr_after_shift(rows: List[Dict[str, Any]], shift_time: int, w: int = 500) -> float:
    if not rows:
        return float("nan")

    start = max(0, shift_time)
    end = min(len(rows), start + w)
    if end <= start:
        return float("nan")

    window = rows[start:end]
    clicks = sum(int(row["reward"]) for row in window)
    return float(clicks / len(window))
