"""What COSA believes is active: the working set, and how to read it out loud.

§3.2 (``paper.tex:268``) says the working set holds three kinds of item at once:

1. active linear inequalities;
2. equality constraints;
3. the currently active geometry of the second-order cone.

:class:`WorkingSet` is that container, and it holds all three -- but not in the same way,
because they are not the same kind of thing:

* **Inequalities** are a choice. Any subset of the rows of ``A`` can be in the set, and
  §7.1 and §7.2 (``paper.tex:569``) are the rules that put them in and take them out.
  Stored as a sorted tuple of row indices, sorted so that the row order of the KKT system
  assembled from the set is a function of the set alone -- two paths that arrive at the
  same active set must produce the same matrix, or the factorization reuse of #27 is
  comparing incomparable things.
* **Equalities** are not a choice. §3.1 (``paper.tex:236``) imposes ``E @ p = 0`` at every
  iterate, unconditionally: an equality can never be dropped, and there is no rule
  anywhere in the plan that would drop one. Storing them as a mutable subset would invite
  code that pretends otherwise, so they are held as a count and exposed through
  :attr:`WorkingSet.equalities`. They are in the set, always, all of them.
* **Cone geometry** is neither a subset nor a constant. A cone contributes rows according
  to *where the iterate sits on it*, and there are three cases, not two -- see
  :class:`ConeStatus`. This is the substance of the project: a polyhedral working set
  records which of finitely many constraints is tight, and a conic one has to record a
  continuum of possible active faces.

The set carries the *shape* of the problem it belongs to -- the number of inequality rows,
the number of equality rows, the cone product -- and not the problem itself. That is what
makes it reusable across the frontier sequence of #30, where a warm start hands the working
set of one ``lambda`` to the problem of the next: same shape, different data.

**Reading the set out loud is a product requirement.** Success Criterion 3
(``paper.tex:1324``) asks that COSA's "working-set decisions can be interpreted in terms
of the active portfolio constraints and SOC geometry", which is a statement about output,
not about debugging. :meth:`WorkingSet.describe` is that output, and
:class:`ConstraintNames` is how a caller gets *budget* and *cap on AAPL* into it instead
of *inequality 0* and *inequality 7*.

Nothing here mutates anything: the class is frozen, and every rule that changes a working
set lives in :mod:`cosa.active_set.updates` and returns a new one.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cosa.problem.socp import ConeProduct, ProblemError

if TYPE_CHECKING:
    from cosa.problem.socp import SOCP, SecondOrderCone

__all__ = [
    "ConeStatus",
    "ConstraintNames",
    "WorkingSet",
]


class ConeStatus(enum.Enum):
    """The active geometry of one second-order cone -- a decision, not an observation.

    Where the iterate *is* on the cone is a fact, reported by
    :class:`cosa.geometry.soc.ConePosition`. What the working set *does* about it is this,
    and the two are deliberately different types. §7.3 (``paper.tex:592``) is explicit
    that "geometric activity alone is not sufficient", and §7.4 hands the deactivation
    decision to the conic multiplier and the normal cone rather than to the geometry, so a
    status is a belief the solver holds and revises, not a lookup.

    Three cases rather than two, which is the whole difference from a polyhedral working
    set. A linear inequality is in or out; a cone can be off, on at a smooth boundary
    point where a tangent hyperplane is the local analogue of an active constraint, or on
    at the apex, where §8.1 (``paper.tex:623``) says there is no tangent hyperplane to use.

    Attributes:
        INACTIVE: the cone constrains nothing locally and contributes no rows. The
            iterate is strictly inside it, or the multiplier says the cone carries no
            genuine active normal.
        TANGENT: active at a nonzero boundary point. Contributes the single row of
            eq. (3), ``tau - u.T @ L @ p = 0`` with ``u = L @ x / ||L @ x||_2``.
        APEX: active at the apex ``(0, 0)``. The tangent representation does not apply,
            so the block is held exactly rather than linearized.
    """

    INACTIVE = "inactive"
    TANGENT = "tangent"
    APEX = "apex"

    @property
    def is_active(self) -> bool:
        """Does this status put the cone in the working set at all?"""
        return self is not ConeStatus.INACTIVE

    @property
    def label(self) -> str:
        """A phrase describing the status, for :meth:`WorkingSet.describe`."""
        return {
            ConeStatus.INACTIVE: "inactive, strictly inside the cone",
            ConeStatus.TANGENT: "active on the boundary, as a tangent hyperplane",
            ConeStatus.APEX: "active at the apex, held exactly",
        }[self]

    def num_rows(self, cone: SecondOrderCone) -> int:
        """How many rows this status contributes to the direction subproblem.

        One for :attr:`TANGENT`: eq. (3) is a single scalar equation, which is exactly
        what makes the tangent representation attractive -- an active cone costs no more
        than an active linear constraint.

        The full block dimension for :attr:`APEX`, because holding the iterate at the apex
        means ``(tau, L @ p) = 0``, and there is no hyperplane that expresses that. The
        count is stated here so that the KKT assembly of #12 can size its system from the
        working set alone; the rows themselves, and the normal-cone conditions behind
        them, belong to #24.

        Args:
            cone: the factor this status describes, whose dimension the apex case needs.

        Returns:
            The number of rows, ``0`` when inactive.
        """
        match self:
            case ConeStatus.INACTIVE:
                return 0
            case ConeStatus.TANGENT:
                return 1
            case ConeStatus.APEX:
                return cone.dim


@dataclass(frozen=True)
class ConstraintNames:
    """Names for a problem's rows, so a working set can be read in the user's terms.

    Success Criterion 3 asks for working-set decisions interpretable "in terms of the
    active portfolio constraints", and a row index is not that. Supplying names is
    optional and partial: any row without one falls back to its index, so a caller can
    name the two constraints that matter and leave four hundred box bounds unnamed.

    Attributes:
        inequalities: names for the rows of ``A``, in row order.
        equalities: names for the rows of ``E``, in row order.
        cones: names for the factors of the cone product, in factor order. A
            mean-standard-deviation portfolio has one, and it is the risk cone.
    """

    inequalities: tuple[str, ...] = ()
    equalities: tuple[str, ...] = ()
    cones: tuple[str, ...] = ()

    @staticmethod
    def _name(names: tuple[str, ...], kind: str, index: int) -> str:
        """Look a name up, falling back to the kind and the index.

        Args:
            names: the names supplied for this kind of row.
            kind: what to call the row when it has no name.
            index: the row's index.

        Returns:
            ``"name (#index)"`` when a name was supplied, ``"kind #index"`` otherwise.
        """
        if index < len(names) and names[index]:
            return f"{names[index]} (#{index})"
        return f"{kind} #{index}"

    def inequality(self, index: int) -> str:
        """The name of inequality row ``index``, or a fallback.

        Args:
            index: the row's index.

        Returns:
            A human-readable name.
        """
        return self._name(self.inequalities, "inequality", index)

    def equality(self, index: int) -> str:
        """The name of equality row ``index``, or a fallback.

        Args:
            index: the row's index.

        Returns:
            A human-readable name.
        """
        return self._name(self.equalities, "equality", index)

    def cone(self, index: int) -> str:
        """The name of cone factor ``index``, or a fallback.

        Args:
            index: the factor's index.

        Returns:
            A human-readable name.
        """
        return self._name(self.cones, "cone", index)


@dataclass(frozen=True)
class WorkingSet:
    """The three item classes of §3.2, held together and validated on construction.

    Immutable, and cheap to copy: the inequality indices and the cone statuses are small
    tuples. Every rule that changes a set is a function in
    :mod:`cosa.active_set.updates` returning a new one, so a solver iteration can keep the
    previous set for free -- which is what anti-cycling detection (#29) and warm starting
    (#30) both need.

    Comparison is structural, unlike :class:`cosa.SOCP`: two working sets are equal when
    they name the same rows and the same cone geometry over the same shape, which is
    exactly the question "did this iteration change the active set?" that the solver loop
    asks. That also makes an instance hashable, so a sequence of visited sets can go in a
    set -- the cycling check of #29.

    Attributes:
        num_inequalities: the number of rows of ``A`` in the problem, active or not.
        num_equalities: the number of rows of ``E``, all of which are always in the set.
        cone: the problem's cone product, whose factor count fixes the length of
            :attr:`cone_status`.
        inequalities: the indices of the active rows of ``A``, ascending and distinct.
        cone_status: one status per factor of :attr:`cone`, in factor order.
    """

    num_inequalities: int
    num_equalities: int
    cone: ConeProduct = field(default_factory=ConeProduct)
    inequalities: tuple[int, ...] = ()
    cone_status: tuple[ConeStatus, ...] = ()

    def __post_init__(self) -> None:
        """Normalize the index tuple and check every part against the shape.

        The inequality indices are sorted rather than required to arrive sorted: the
        ascending order is what makes the assembled KKT row order a function of the set,
        and enforcing it here is more useful than rejecting a caller who built the tuple
        in discovery order. Repeats are rejected, because a repeated index means the
        caller believes a row can be active twice.

        Raises:
            ProblemError: if a row count is negative, if an inequality index is out of
                range or repeated, or if there is not exactly one status per cone factor.
        """
        set_field = object.__setattr__
        for name, count in (("num_inequalities", self.num_inequalities), ("num_equalities", self.num_equalities)):
            if count < 0:
                raise ProblemError(name, f"a row count is non-negative, found {count}")

        indices = tuple(sorted(int(index) for index in self.inequalities))
        if len(set(indices)) != len(indices):
            raise ProblemError("inequalities", f"a row cannot be active twice, found {self.inequalities}")
        for index in indices:
            self._check_inequality(index)
        set_field(self, "inequalities", indices)

        statuses = tuple(self.cone_status)
        if len(statuses) != self.num_cones:
            raise ProblemError(
                "cone_status",
                f"expected one status per cone factor, so {self.num_cones}, found {len(statuses)}",
            )
        for index, status in enumerate(statuses):
            if not isinstance(status, ConeStatus):
                raise ProblemError("cone_status", f"entry {index} is {status!r}, which is not a ConeStatus")
        set_field(self, "cone_status", statuses)

    @classmethod
    def empty(cls, problem: SOCP) -> WorkingSet:
        """The starting working set for a problem: no inequality active, no cone active.

        Where Phase I begins. The equalities are in it, because they always are.

        Args:
            problem: the instance whose shape the set is built for.

        Returns:
            A set over ``problem``'s shape with nothing chosen yet.
        """
        return cls(
            num_inequalities=problem.num_inequalities,
            num_equalities=problem.num_equalities,
            cone=problem.cone,
            inequalities=(),
            cone_status=tuple(ConeStatus.INACTIVE for _ in problem.cone.cones),
        )

    @property
    def num_cones(self) -> int:
        """The number of cone factors the set tracks."""
        return len(self.cone)

    @property
    def equalities(self) -> tuple[int, ...]:
        """Every equality row index, because every equality is always in the set.

        A property rather than a field: §3.1 imposes ``E @ p = 0`` unconditionally, so
        there is no state here to store and no rule that could change it.
        """
        return tuple(range(self.num_equalities))

    @property
    def inactive_inequalities(self) -> tuple[int, ...]:
        """The inequality rows not in the set -- the candidates §7.1 can add."""
        active = set(self.inequalities)
        return tuple(index for index in range(self.num_inequalities) if index not in active)

    @property
    def active_cones(self) -> tuple[int, ...]:
        """The indices of the cone factors whose status is not inactive."""
        return tuple(index for index, status in enumerate(self.cone_status) if status.is_active)

    @property
    def num_rows(self) -> int:
        """The number of rows the set imposes on the direction subproblem.

        The active inequalities, plus every equality, plus each active cone's
        contribution per :meth:`ConeStatus.num_rows`. The size of the ``W`` block of the
        KKT system of §13, which is why it is a property of the working set rather than
        something the assembly counts for itself.
        """
        conic = sum(status.num_rows(cone) for status, cone in zip(self.cone_status, self.cone.cones, strict=True))
        return len(self.inequalities) + self.num_equalities + conic

    def is_active(self, index: int) -> bool:
        """Is inequality row ``index`` in the set?

        Args:
            index: the row's index.

        Returns:
            ``True`` if the row is active.

        Raises:
            ProblemError: if the index is not an inequality row of this shape.
        """
        self._check_inequality(index)
        return index in self.inequalities

    def status(self, index: int) -> ConeStatus:
        """The status of cone factor ``index``.

        Args:
            index: the factor's index.

        Returns:
            Its status.

        Raises:
            ProblemError: if the index is not a factor of this shape's cone product.
        """
        self._check_cone(index)
        return self.cone_status[index]

    def _check_inequality(self, index: int) -> None:
        """Reject an index that is not an inequality row of this shape.

        Args:
            index: the index to check.

        Raises:
            ProblemError: if the index is out of range.
        """
        if not 0 <= index < self.num_inequalities:
            raise ProblemError(
                "inequality",
                f"expected an index in [0, {self.num_inequalities}), found {index}",
            )

    def _check_cone(self, index: int) -> None:
        """Reject an index that is not a cone factor of this shape.

        Args:
            index: the index to check.

        Raises:
            ProblemError: if the index is out of range.
        """
        if not 0 <= index < self.num_cones:
            raise ProblemError(
                "cone",
                f"expected an index in [0, {self.num_cones}), found {index}",
            )

    def describe(self, names: ConstraintNames | None = None) -> str:
        """The working set in words, in terms of the named constraints and the geometry.

        Success Criterion 3, made executable. The rendering is deterministic and stable
        across runs -- rows appear in index order -- so two iterates' descriptions can be
        diffed to see what a step decided.

        Args:
            names: names for the problem's rows, or ``None`` to name every row by its
                index.

        Returns:
            A multi-line description, with no trailing newline.
        """
        labels = names or ConstraintNames()
        lines = [
            f"working set: {self.num_rows} row(s) over "
            f"{self.num_inequalities} inequality(-ies), "
            f"{self.num_equalities} equality(-ies), "
            f"{self.num_cones} cone(s)"
        ]
        lines.append(self._render("active inequalities", [labels.inequality(i) for i in self.inequalities]))
        lines.append(self._render("inactive inequalities", [labels.inequality(i) for i in self.inactive_inequalities]))
        lines.append(self._render("equalities, always active", [labels.equality(i) for i in self.equalities]))
        if self.num_cones:
            lines.append("  cone geometry:")
            for index, status in enumerate(self.cone_status):
                rows = status.num_rows(self.cone.cones[index])
                lines.append(f"    {labels.cone(index)}: {status.label} ({rows} row(s))")
        else:
            lines.append("  cone geometry: none -- this instance is a linear program")
        return "\n".join(lines)

    @staticmethod
    def _render(heading: str, entries: list[str]) -> str:
        """Render one heading and its entries as a single indented line.

        Args:
            heading: what the entries are.
            entries: the entries, already named.

        Returns:
            The line, with ``"none"`` where there are no entries.
        """
        return f"  {heading}: {', '.join(entries) if entries else 'none'}"

    def __str__(self) -> str:
        """The description with default names -- what a traceback or a log line shows."""
        return self.describe()
