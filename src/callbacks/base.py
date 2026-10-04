"""Callback interface.

A small, explicit callback system (plan section 8.3: the Stage 1 -> Stage 2
transition "must be managed by a Callback / Training Controller").

Available hooks
---------------
``on_train_begin``   once, before stage 1
``on_stage_begin``   at the start of every stage (optimizer/loss are rebuilt)
``on_epoch_begin``   before each epoch
``on_step_end``      after every optimizer step
``on_epoch_end``     after validation, with the epoch metrics in ``state``
``on_stage_end``     after the last epoch of a stage
``on_train_end``     once, after the final stage

Callbacks must not raise on unknown hooks, so adding a new hook stays backwards
compatible.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

HOOKS = (
    "on_train_begin",
    "on_stage_begin",
    "on_epoch_begin",
    "on_step_end",
    "on_epoch_end",
    "on_stage_end",
    "on_train_end",
)


class Callback:
    """Base class - override the hooks you need."""

    name: str = "callback"

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    # hooks ---------------------------------------------------------------- #
    def on_train_begin(self, trainer: Any = None, **kwargs: Any) -> None: ...

    def on_stage_begin(self, trainer: Any = None, stage: Any = None, **kwargs: Any) -> None: ...

    def on_epoch_begin(self, trainer: Any = None, epoch: int = 0, stage: Any = None, **kwargs: Any) -> None: ...

    def on_step_end(self, trainer: Any = None, step: int = 0, **kwargs: Any) -> None: ...

    def on_epoch_end(self, trainer: Any = None, epoch: int = 0, metrics: Dict[str, Any] | None = None,
                     stage: Any = None, **kwargs: Any) -> None: ...

    def on_stage_end(self, trainer: Any = None, stage: Any = None, **kwargs: Any) -> None: ...

    def on_train_end(self, trainer: Any = None, **kwargs: Any) -> None: ...

    # helpers -------------------------------------------------------------- #
    def set_trainer(self, trainer: Any) -> None:
        self.trainer = trainer

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}()"


class CallbackList:
    """Fan a hook out to a list of callbacks; keeps declaration order."""

    def __init__(self, callbacks: Sequence[Callback] | None = None) -> None:
        self.callbacks: List[Callback] = list(callbacks or [])

    def add(self, callback: Callback) -> "CallbackList":
        self.callbacks.append(callback)
        return self

    def extend(self, callbacks: Sequence[Callback]) -> "CallbackList":
        self.callbacks.extend(callbacks)
        return self

    def _dispatch(self, hook: str, **kwargs: Any) -> None:
        for cb in self.callbacks:
            fn = getattr(cb, hook, None)
            if callable(fn):
                fn(**kwargs)

    # public API mirroring Callback --------------------------------------- #
    def on_train_begin(self, **kwargs: Any) -> None:
        self._dispatch("on_train_begin", **kwargs)

    def on_stage_begin(self, **kwargs: Any) -> None:
        self._dispatch("on_stage_begin", **kwargs)

    def on_epoch_begin(self, **kwargs: Any) -> None:
        self._dispatch("on_epoch_begin", **kwargs)

    def on_step_end(self, **kwargs: Any) -> None:
        self._dispatch("on_step_end", **kwargs)

    def on_epoch_end(self, **kwargs: Any) -> None:
        self._dispatch("on_epoch_end", **kwargs)

    def on_stage_end(self, **kwargs: Any) -> None:
        self._dispatch("on_stage_end", **kwargs)

    def on_train_end(self, **kwargs: Any) -> None:
        self._dispatch("on_train_end", **kwargs)

    def __len__(self) -> int:
        return len(self.callbacks)

    def __iter__(self):
        return iter(self.callbacks)

    def describe(self) -> List[str]:
        return [type(cb).__name__ for cb in self.callbacks]


__all__ = ["HOOKS", "Callback", "CallbackList"]
