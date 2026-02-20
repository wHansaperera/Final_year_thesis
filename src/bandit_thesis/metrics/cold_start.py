# from __future__ import annotations
# from typing import Any, Dict, List
# import numpy as np


# def cold_start_ctr_per_user(rows: List[Dict[str, Any]], m: int = 30) -> float:
#     """
#     Cold-start CTR averaged over users, using each user's first m impressions.

#     rows: list of dicts with keys:
#       - "reward" (0/1)
#       - "context" containing "user_id"
#       - optional "t" etc.

#     Returns: mean_u ( clicks_u_first_m / m ) over users with >= m impressions.
#     """
#     user_counts: Dict[int, int] = {}
#     user_clicks_first_m: Dict[int, int] = {}

#     for r in rows:
#         uid = int(r["context"]["user_id"])
#         y = int(r["reward"])
#         c = user_counts.get(uid, 0)

#         if c < m:
#             user_clicks_first_m[uid] = user_clicks_first_m.get(uid, 0) + y

#         user_counts[uid] = c + 1

#     eligible = [uid for uid, cnt in user_counts.items() if cnt >= m]
#     if not eligible:
#         return float("nan")

#     vals = [(user_clicks_first_m.get(uid, 0) / m) for uid in eligible]
#     return float(np.mean(vals))

# def cold_start_regret_per_user(rows: List[Dict[str, Any]], m: int = 30) -> float:
#     """
#     Mean cumulative pseudo-regret over each user's first m impressions,
#     averaged over users with >= m impressions.
#     """
#     user_counts: Dict[int, int] = {}
#     user_regret_first_m: Dict[int, float] = {}

#     for r in rows:
#         uid = int(r["context"]["user_id"])
#         c = user_counts.get(uid, 0)

#         if c < m:
#             reg = float(r["p_opt"]) - float(r["p_chosen"])
#             user_regret_first_m[uid] = user_regret_first_m.get(uid, 0.0) + reg

#         user_counts[uid] = c + 1

#     eligible = [uid for uid, cnt in user_counts.items() if cnt >= m]
#     if not eligible:
#         return float("nan")

#     vals = [user_regret_first_m.get(uid, 0.0) for uid in eligible]
#     return float(np.mean(vals))


# def ctr_at_n(rows: List[Dict[str, Any]], n: int = 1000) -> float:
#     """CTR over the first n global steps."""
#     n = min(n, len(rows))
#     if n == 0:
#         return float("nan")
#     clicks = sum(int(r["reward"]) for r in rows[:n])
#     return float(clicks / n)

# def regret_at_n(rows: List[Dict[str, Any]], n: int = 1000) -> float:
#     """Cumulative pseudo-regret over the first n steps."""
#     n = min(n, len(rows))
#     if n == 0:
#         return float("nan")
#     return float(sum(float(r["p_opt"]) - float(r["p_chosen"]) for r in rows[:n]))

#-----------------------------------------------------------------------------------------#

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np


def cold_start_ctr_per_user(rows: List[Dict[str, Any]], m: int = 30) -> float:
    """
    Cold-start CTR averaged over users using each user's first m impressions.
    Requires rows[*]["context"]["user_id"] and rows[*]["reward"].
    Only users with >= m impressions are included.
    """
    user_counts: Dict[int, int] = {}
    user_clicks_first_m: Dict[int, int] = {}

    for r in rows:
        uid = int(r["context"]["user_id"])
        y = int(r["reward"])
        c = user_counts.get(uid, 0)

        if c < m:
            user_clicks_first_m[uid] = user_clicks_first_m.get(uid, 0) + y

        user_counts[uid] = c + 1

    eligible = [uid for uid, cnt in user_counts.items() if cnt >= m]
    if not eligible:
        return float("nan")

    vals = [user_clicks_first_m.get(uid, 0) / m for uid in eligible]
    return float(np.nanmean(vals))


def cold_start_regret_per_user(rows: List[Dict[str, Any]], m: int = 30) -> float:
    """
    Cold-start pseudo-regret averaged over users using each user's first m impressions.
    Requires rows[*]["context"]["user_id"], rows[*]["p_opt"], rows[*]["p_chosen"].
    Only users with >= m impressions are included.

    Returns mean_u sum_{i=1..m}(p_opt - p_chosen).
    """
    user_counts: Dict[int, int] = {}
    user_regret_first_m: Dict[int, float] = {}

    for r in rows:
        uid = int(r["context"]["user_id"])
        c = user_counts.get(uid, 0)

        if c < m:
            reg = float(r["p_opt"]) - float(r["p_chosen"])
            user_regret_first_m[uid] = user_regret_first_m.get(uid, 0.0) + reg

        user_counts[uid] = c + 1

    eligible = [uid for uid, cnt in user_counts.items() if cnt >= m]
    if not eligible:
        return float("nan")

    vals = [user_regret_first_m.get(uid, 0.0) for uid in eligible]
    return float(np.nanmean(vals))


def ctr_at_n(rows: List[Dict[str, Any]], n: int = 1000) -> float:
    """CTR over the first n global steps."""
    n = min(n, len(rows))
    if n <= 0:
        return float("nan")
    clicks = sum(int(r["reward"]) for r in rows[:n])
    return float(clicks / n)


def regret_at_n(rows: List[Dict[str, Any]], n: int = 1000) -> float:
    """Cumulative pseudo-regret over the first n global steps."""
    n = min(n, len(rows))
    if n <= 0:
        return float("nan")
    return float(sum(float(r["p_opt"]) - float(r["p_chosen"]) for r in rows[:n]))


def ctr_last_w(rows: List[Dict[str, Any]], w: int = 1000) -> float:
    """
    CTR over the last w steps.
    Very useful for non-stationary adaptation: higher is better.
    """
    if not rows:
        return float("nan")
    w = min(w, len(rows))
    tail = rows[-w:]
    clicks = sum(int(r["reward"]) for r in tail)
    return float(clicks / w)


def regret_last_w(rows: List[Dict[str, Any]], w: int = 1000) -> float:
    """
    Cumulative pseudo-regret over the last w steps.
    Useful in non-stationary settings to measure recent regret.
    """
    if not rows:
        return float("nan")
    w = min(w, len(rows))
    tail = rows[-w:]
    return float(sum(float(r["p_opt"]) - float(r["p_chosen"]) for r in tail))


def ctr_after_shift(rows: List[Dict[str, Any]], shift_time: int, w: int = 1000) -> float:
    """
    CTR in the window immediately AFTER the shift.
    Example: shift_time=5000, w=1000 -> computes CTR on steps [5000..5999] (if available).

    This is the cleanest metric to show "recovery" after non-stationary change.
    """
    if not rows:
        return float("nan")

    start = max(0, shift_time)
    end = min(len(rows), start + w)
    if end <= start:
        return float("nan")

    window = rows[start:end]
    clicks = sum(int(r["reward"]) for r in window)
    return float(clicks / (end - start))