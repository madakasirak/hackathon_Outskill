"""Tests for Pydantic schemas. No LLM, no mocks — these run instantly and prove the contracts."""
import pytest
from pydantic import ValidationError

# These imports will resolve once the team refactors agents.py into core/schemas.py.
# Until then, you can run the same tests interactively against the schemas defined
# in the notebook by importing them directly.
pytest.importorskip("core.schemas", reason="Run this once core/schemas.py exists")

from core.schemas import (  # noqa: E402
    ChartSpec,
    FactCheckResult,
    ReflectionResult,
    ResearchPlan,
    SourceAssessment,
    SourceTrust,
    SubQuestion,
)


class TestResearchPlan:
    def test_valid_plan_passes(self):
        plan = ResearchPlan(
            sub_questions=[
                SubQuestion(question="What is X?", rationale="Foundational"),
                SubQuestion(question="How does Y?", rationale="Mechanism"),
            ],
            key_concepts=["X", "Y", "Z"],
        )
        assert len(plan.sub_questions) == 2
        assert "X" in plan.key_concepts

    def test_empty_sub_questions_allowed(self):
        # Edge case: planner can return a plan with no sub-questions if it fails
        plan = ResearchPlan(sub_questions=[], key_concepts=[])
        assert plan.sub_questions == []


class TestSourceTrust:
    def test_valid_trust_levels(self):
        for level in ("high", "medium", "low"):
            t = SourceTrust(source_id="example.com", trust_score=level, rationale="...")
            assert t.trust_score == level

    def test_invalid_trust_level_rejected(self):
        with pytest.raises(ValidationError):
            SourceTrust(source_id="x", trust_score="extreme", rationale="...")

    def test_source_assessment_aggregates(self):
        a = SourceAssessment(assessments=[
            SourceTrust(source_id="a", trust_score="high", rationale="r"),
            SourceTrust(source_id="b", trust_score="low", rationale="r"),
        ])
        high_count = sum(1 for x in a.assessments if x.trust_score == "high")
        assert high_count == 1


class TestFactCheckResult:
    def test_valid_verdicts(self):
        for v in ("supported", "unsupported", "contradicted"):
            fc = FactCheckResult(claim="X is true", verdict=v, evidence="...")
            assert fc.verdict == v

    def test_invalid_verdict_rejected(self):
        with pytest.raises(ValidationError):
            FactCheckResult(claim="X", verdict="maybe", evidence="...")


class TestReflectionResult:
    def test_full_reflection(self):
        r = ReflectionResult(
            coverage_gaps=["What about rural areas?"],
            contradictions=["Source A says X, Source B says not X"],
            overlooked_angles=["Long-term environmental impact"],
            needs_more_research=True,
        )
        assert r.needs_more_research is True
        assert len(r.contradictions) == 1

    def test_empty_reflection_means_done(self):
        r = ReflectionResult(
            coverage_gaps=[],
            contradictions=[],
            overlooked_angles=[],
            needs_more_research=False,
        )
        assert r.needs_more_research is False


class TestChartSpec:
    def test_bar_chart_spec(self):
        spec = ChartSpec(
            chart_type="bar",
            title="Adoption by year",
            x_labels=["2022", "2023", "2024"],
            y_values=[2.1, 4.5, 8.0],
            y_label="% adoption",
        )
        assert spec.chart_type == "bar"
        assert len(spec.x_labels) == len(spec.y_values)

    def test_none_chart_for_no_data(self):
        spec = ChartSpec(chart_type="none", title="", x_labels=[], y_values=[])
        assert spec.chart_type == "none"

    def test_invalid_chart_type_rejected(self):
        with pytest.raises(ValidationError):
            ChartSpec(chart_type="3d-hologram", title="", x_labels=[], y_values=[])
