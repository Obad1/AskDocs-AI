"""SM-2 spaced-repetition scheduler (spec §7.3 ReviewFlashcard / §3.6 Study).

Pure, side-effect-free implementation that mirrors the frontend WorkspaceContext
SM-2 step, so the server can independently schedule flashcards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Rating mapping per spec §3.6: Again=0, Hard=3, Good=4, Easy=5.
RATING_AGAIN = 0
RATING_HARD = 3
RATING_GOOD = 4
RATING_EASY = 5

MIN_EASINESS = 1.3
DEFAULT_EASINESS = 2.5
DAY_MS = 86_400_000


@dataclass
class SM2State:
    repetitions: int = 0
    interval: int = 0  # days
    easiness: float = DEFAULT_EASINESS
    due_date: int = 0  # epoch ms


def review(card: SM2State, rating: int, now: int) -> SM2State:
    """Apply one SM-2 review step. ``rating`` is 0..5 (see RATING_* constants)."""
    if card.repetitions == 0 and card.interval == 0 and card.due_date == 0:
        # Fresh card: initialize due date to now so it is immediately due.
        card = SM2State(due_date=now)

    ease = card.easiness
    new_ease = max(
        MIN_EASINESS,
        ease + (0.1 - (5 - rating) * (0.08 + (5 - rating) * 0.02)),
    )

    reps = card.repetitions
    interval = card.interval
    if rating < 3:  # Again
        new_reps = 0
        new_interval = 1
    elif reps == 0:
        new_reps = 1
        new_interval = 1
    elif reps == 1:
        new_reps = 2
        new_interval = 6
    else:
        new_reps = reps + 1
        new_interval = max(1, round(interval * new_ease))

    due = now + new_interval * DAY_MS
    return SM2State(
        repetitions=new_reps,
        interval=new_interval,
        easiness=new_ease,
        due_date=due,
    )


def is_due(card: SM2State, now: Optional[int] = None) -> bool:
    import time

    return card.due_date <= (now if now is not None else int(time.time() * 1000))
