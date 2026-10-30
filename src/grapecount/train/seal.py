"""A runtime seal over the outer-test set.

The split already makes selection-on-outer-test hard: an outer-test bunch has no row in
`bunch_inner.tsv`, so `inner_split` cannot return it. That is a structural guarantee about
the *data*. This adds a structural guarantee about the *control flow*: the outer-test
bunches are wrapped in an object that raises if anything reads them before selection has
been declared finished.

Why bother, given the split already separates them. Because the typical failure is not
someone deliberately evaluating on the test fold -- it is a training loop that happens to
have the reported fold in scope and uses it for early stopping, unnoticed. A seal turns that class of accident from a silent bias into a stack trace.

    seal = SealedBunches(split.outer_test(o))
    ...                                  # inner CV; touching `seal` here raises
    epoch = select(inner_scores)         # decided
    seal.unseal("selection complete")    # explicit, recorded, once
    test_bunches = seal.get()            # only now
"""

from __future__ import annotations

from dataclasses import dataclass, field


class SealViolation(RuntimeError):
    """Raised when sealed data is read before selection has been declared complete."""


@dataclass
class SealedBunches:
    """Outer-test bunch labels, readable only after an explicit unseal.

    `len()` and membership are deliberately NOT provided while sealed: knowing the size or
    testing membership of the test set is already enough to leak, e.g. by filtering training
    data against it.
    """

    _bunches: frozenset[str]
    _unsealed: bool = False
    _reason: str = ""
    _access_log: list[str] = field(default_factory=list)

    def __init__(self, bunches: set[str] | frozenset[str]) -> None:
        object.__setattr__(self, "_bunches", frozenset(bunches))
        object.__setattr__(self, "_unsealed", False)
        object.__setattr__(self, "_reason", "")
        object.__setattr__(self, "_access_log", [])

    # ------------------------------------------------------------------ access
    def get(self) -> frozenset[str]:
        if not self._unsealed:
            raise SealViolation(
                "outer-test bunches were read while still sealed. Model selection must be "
                "complete, and unseal() called with a reason, before the test set is "
                "touched. If this fired inside a training or epoch-selection routine, that "
                "routine is the bug -- selecting on the test fold biases the reported "
                "score optimistically."
            )
        self._access_log.append(self._reason)
        return self._bunches

    def unseal(self, reason: str) -> None:
        if self._unsealed:
            raise SealViolation(
                f"already unsealed for {self._reason!r}; a fold is scored once, not twice"
            )
        if not reason:
            raise ValueError("unseal() requires a reason, which is recorded in the run manifest")
        object.__setattr__(self, "_unsealed", True)
        object.__setattr__(self, "_reason", reason)

    # --------------------------------------------------------------- introspection
    @property
    def is_sealed(self) -> bool:
        return not self._unsealed

    @property
    def n_accesses(self) -> int:
        return len(self._access_log)

    def audit(self) -> dict:
        return {
            "sealed": not self._unsealed,
            "unseal_reason": self._reason,
            "n_accesses": len(self._access_log),
            "n_bunches": len(self._bunches) if self._unsealed else None,
        }

    # Deny the sneaky routes. Each of these would otherwise leak the test set silently.
    def __len__(self) -> int:
        raise SealViolation("len() of the outer-test set is itself information; call get()")

    def __contains__(self, item: object) -> bool:
        raise SealViolation("membership test against the outer-test set; call get()")

    def __iter__(self):
        raise SealViolation("iteration over the outer-test set; call get()")

    def __repr__(self) -> str:
        return f"SealedBunches(sealed={not self._unsealed})"
