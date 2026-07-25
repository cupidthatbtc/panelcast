"""claims.yaml — the machine-checkable replication claim spec (#272).

One file per domain declares the paper's quantitative claims as assertions
over posterior quantities:

    claims:
      - name: cohort_improvement
        quantity: group_mean_trend
        expect: {direction: increasing, from: "1900s"}
      - name: peak_age
        quantity: covariate_vertex(age_c, age_sq)
        expect: {in: [30, 40]}
        grade: qualitative
      - name: elite_premium
        quantity: entity_contrast
        entities: {group_a: [Kasparov, Carlsen], group_b: rest}
        expect: {greater_than: 0, prob: 0.95}

``quantity`` is a named extractor, optionally with positional arguments in
parentheses. ``expect`` grades against posterior draws, never point
estimates. ``grade`` is the claim's target rung on the ladder
(match > qualitative > shape_only); failing the target while passing a lower
rung is a *divergence*, not an error — divergences under a fixed model are
findings.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_QUANTITY_RE = re.compile(r"^(?P<name>[a-z_][a-z0-9_]*)(?:\((?P<args>[^)]*)\))?$")

GRADES = ("match", "qualitative", "shape_only")


class ExpectSpec(BaseModel):
    """What the paper claims about the quantity, graded against draws."""

    model_config = ConfigDict(extra="forbid")

    direction: Literal["increasing", "decreasing"] | None = None
    # Alias "in" is the natural YAML key; the attribute avoids the keyword.
    within: tuple[float, float] | None = Field(default=None, alias="in")
    greater_than: float | None = None
    less_than: float | None = None
    # Posterior probability the assertion must reach to count as met.
    prob: float = Field(default=0.9, gt=0.5, le=1.0)
    # Optional anchor for direction claims (e.g. trend measured from "1900s").
    from_: str | None = Field(default=None, alias="from")

    @model_validator(mode="after")
    def _one_assertion(self) -> ExpectSpec:
        assertions = [
            self.direction is not None,
            self.within is not None,
            self.greater_than is not None,
            self.less_than is not None,
        ]
        if sum(assertions) != 1:
            raise ValueError(
                "expect must declare exactly one of: direction, in, "
                "greater_than, less_than."
            )
        if self.within is not None and not self.within[0] < self.within[1]:
            raise ValueError(f"expect.in must be [low, high], got {list(self.within)}.")
        return self


class EntitySets(BaseModel):
    """Explicit entity sets for contrast/ranking claims."""

    model_config = ConfigDict(extra="forbid")

    group_a: list[str]
    # "rest" contrasts group_a against every other entity.
    group_b: list[str] | Literal["rest"] = "rest"


class ClaimSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    quantity: str
    expect: ExpectSpec
    grade: Literal["match", "qualitative", "shape_only"] = "match"
    entities: EntitySets | None = None

    @property
    def extractor_name(self) -> str:
        match = _QUANTITY_RE.match(self.quantity.strip())
        if match is None:
            raise ValueError(f"claim '{self.name}': unparseable quantity '{self.quantity}'.")
        return match.group("name")

    @property
    def extractor_args(self) -> list[str]:
        match = _QUANTITY_RE.match(self.quantity.strip())
        if match is None:
            raise ValueError(f"claim '{self.name}': unparseable quantity '{self.quantity}'.")
        raw = match.group("args")
        if not raw:
            return []
        return [a.strip() for a in raw.split(",") if a.strip()]

    @model_validator(mode="after")
    def _quantity_parses(self) -> ClaimSpec:
        _ = self.extractor_name
        return self


class ClaimsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[ClaimSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> ClaimsFile:
        names = [c.name for c in self.claims]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate claim names: {sorted(dupes)}.")
        return self


def load_claims(path: Path | str) -> ClaimsFile:
    """Parse and validate a claims.yaml. Raises on unknown keys or bad specs."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping with a 'claims' list.")
    return ClaimsFile(**data)
