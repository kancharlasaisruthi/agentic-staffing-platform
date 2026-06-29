"""
Pydantic models for the ICP / run configuration.

This is the ONLY layer the rest of the platform should trust for
domain-specific values (industry, persona names, thresholds, etc).
Agents must never hardcode a persona name, industry, or threshold —
they read it from a validated Config instance passed through state.
"""

from __future__ import annotations

from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ICPConfig(BaseModel):
    employee_size_min: int = Field(ge=1, description="Minimum employee count to qualify as ICP")
    employee_size_max: Optional[int] = Field(default=None, description="Optional upper bound")
    locations: list[str] = Field(default_factory=lambda: ["United States"])
    hiring_focus: list[str] = Field(
        default_factory=list,
        description="Role families this run cares about, e.g. 'Backend Engineers'",
    )

    @field_validator("hiring_focus")
    @classmethod
    def _non_empty_focus(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("icp.hiring_focus must list at least one role family")
        return v


class RunConfig(BaseModel):
    industry: str
    icp: ICPConfig
    hiring_threshold: int = Field(ge=0, description="Minimum estimated open roles to qualify a company")
    target_personas: list[str] = Field(
        default_factory=list,
        description="Ranked list of decision-maker titles to search for, in priority order",
    )
    max_companies: int = Field(default=15, ge=1, le=100, description="Cap on companies processed per run")
    search_depth: str = Field(default="basic", description="Tavily search_depth: 'basic' or 'advanced'")

    @field_validator("target_personas")
    @classmethod
    def _non_empty_personas(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("target_personas must list at least one persona")
        return v

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "RunConfig":
        return cls(**raw)
