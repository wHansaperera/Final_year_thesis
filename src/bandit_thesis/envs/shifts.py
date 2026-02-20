from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


@dataclass
class AbruptShift:
    """At shift_time, call on_shift(env) once."""
    shift_time: int
    on_shift: Callable[[object], None]
    _done: bool = False

    def apply(self, env: object, t: int) -> None:
        if (not self._done) and t >= self.shift_time:
            self.on_shift(env)
            self._done = True
