"""Per-agent unit tests with mocked LLMs.

Pattern: each test patches the LLM globals in core.agents to return canned
Pydantic instances, then asserts the agent's contract (input state → output dict).

These tests will pass once the team refactors the notebook into core/agents.py.
Until then, they serve as a template — copy the structure and import from
wherever your agents live.
"""
import pytest

pytest.importorskip("core.agents", reason="Run after refactoring notebook into core/agents.py")

from langchain_core.documents import Document  # noqa: E402

# Once core/ exists, these imports resolve:
# from core import agents
# from core.schemas import (
#     FactCheckResult, ReflectionResult, ResearchPlan, SourceAssessment,
#     SourceTrust, SubQuestion, ChartSpec,
# )


class TestPlannerAgent:
    def test_planner_produces_plan_with_sub_questions(self, fake_llm, base_state, monkeypatch):
        from core import agents
        from core.schemas import ResearchPlan, SubQuestion

        canned = ResearchPlan(
            sub_questions=[
                SubQuestion(question="Q1?", rationale="R1"),
                SubQuestion(question="Q2?", rationale="R2"),
            ],
            key_concepts=["concept1"],
        )
        fake_llm.invoke.return_value = canned
        monkeypatch.setattr(agents, "reasoning_llm", fake_llm)

        result = agents.planner_agent(base_state)

        assert "plan" in result
        assert len(result["plan"].sub_questions) == 2
        assert result["iteration"] == 0


class TestRetrieverAgent:
    def test_retriever_handles_all_tools_failing_gracefully(
        self, base_state, sample_documents, monkeypatch
    ):
        """Even if every tool raises, retriever should return a valid dict."""
        from core import agents
        from core.schemas import ResearchPlan, SubQuestion

        base_state["plan"] = ResearchPlan(
            sub_questions=[SubQuestion(question="Q?", rationale="R")],
            key_concepts=[],
        )

        # Make every tool raise
        def boom(*a, **kw):
            raise RuntimeError("tool failed")

        monkeypatch.setattr(agents.tavily_tool, "invoke", boom)
        monkeypatch.setattr(agents.arxiv_tool, "invoke", boom)
        monkeypatch.setattr(agents.wiki_tool, "invoke", boom)

        result = agents.retriever_agent(base_state)

        assert "raw_findings" in result
        assert "documents" in result
        assert isinstance(result["raw_findings"], list)


class TestSourceReliabilityAgent:
    def test_reliability_assessment_returned(
        self, fake_llm, base_state, sample_documents, monkeypatch
    ):
        from core import agents
        from core.schemas import SourceAssessment, SourceTrust

        canned = SourceAssessment(assessments=[
            SourceTrust(source_id="https://gov.example/", trust_score="high", rationale="govt"),
            SourceTrust(source_id="https://blog.example/", trust_score="low", rationale="blog"),
        ])
        fake_llm.invoke.return_value = canned
        monkeypatch.setattr(agents, "fast_llm", fake_llm)

        base_state["documents"] = sample_documents
        result = agents.source_reliability_agent(base_state)

        assert "source_assessments" in result
        assert any(a.trust_score == "high" for a in result["source_assessments"])

    def test_reliability_skips_when_no_documents(self, base_state):
        from core import agents
        result = agents.source_reliability_agent(base_state)
        assert result["source_assessments"] == []


class TestAnalyzerAgent:
    def test_analyzer_extracts_insights_from_findings(
        self, fake_llm, base_state, monkeypatch
    ):
        from core import agents

        # Analyzer uses raw text response, not structured output
        fake_llm.invoke.return_value.content = (
            "1. EV adoption is growing\n"
            "2. Battery costs are decreasing\n"
            "3. Infrastructure lags adoption"
        )
        monkeypatch.setattr(agents, "fast_llm", fake_llm)

        base_state["raw_findings"] = ["[Web] Some finding text..."]
        result = agents.analyzer_agent(base_state)

        assert "insights" in result
        assert len(result["insights"]) >= 1


class TestFactCheckerAgent:
    def test_fact_checker_uses_rag(
        self, fake_llm, fake_vector_store, base_state, monkeypatch
    ):
        from core import agents
        from core.schemas import FactCheckResult

        canned = FactCheckResult(
            claim="EV adoption is growing",
            verdict="supported",
            evidence="Government report shows 8% growth.",
        )
        fake_llm.invoke.return_value = canned
        monkeypatch.setattr(agents, "reasoning_llm", fake_llm)
        monkeypatch.setattr(agents, "vector_store", fake_vector_store)

        base_state["insights"] = ["EV adoption is growing"]
        result = agents.fact_checker_agent(base_state)

        # Verify FactChecker queried the RAG store
        fake_vector_store.similarity_search.assert_called()
        assert "fact_checks" in result


class TestReflectionAgent:
    def test_reflection_caps_iterations(self, fake_llm, base_state, monkeypatch):
        from core import agents
        from core.schemas import ReflectionResult, ResearchPlan, SubQuestion

        # Even if the LLM says needs_more_research=True, we should clamp to False
        # when iteration is at the max
        canned = ReflectionResult(
            coverage_gaps=["gap"],
            contradictions=[],
            overlooked_angles=[],
            needs_more_research=True,
        )
        fake_llm.invoke.return_value = canned
        monkeypatch.setattr(agents, "reasoning_llm", fake_llm)

        base_state["plan"] = ResearchPlan(
            sub_questions=[SubQuestion(question="Q?", rationale="R")],
            key_concepts=[],
        )
        base_state["iteration"] = agents.MAX_ITERATIONS - 1  # next bump hits the cap

        result = agents.reflection_agent(base_state)

        assert result["reflection"].needs_more_research is False
        assert result["iteration"] == agents.MAX_ITERATIONS


class TestVisualizationAgent:
    def test_visualization_skips_when_no_chartable_data(
        self, fake_llm, base_state, monkeypatch
    ):
        from core import agents
        from core.schemas import ChartSpec

        canned = ChartSpec(
            chart_type="none", title="", x_labels=[], y_values=[],
        )
        fake_llm.invoke.return_value = canned
        monkeypatch.setattr(agents, "fast_llm", fake_llm)

        result = agents.visualization_agent(base_state)
        assert result["chart_path"] is None


class TestReportBuilderAgent:
    def test_report_builder_emits_markdown_report(
        self, fake_llm, base_state, monkeypatch
    ):
        from core import agents

        fake_llm.invoke.return_value.content = "## Executive Summary\n\nFindings here."
        monkeypatch.setattr(agents, "reasoning_llm", fake_llm)

        result = agents.report_builder_agent(base_state)

        assert "final_report" in result
        assert "## Executive Summary" in result["final_report"]


class TestRouting:
    def test_should_continue_loops_on_gaps(self, base_state):
        from core import agents
        from core.schemas import ReflectionResult

        base_state["reflection"] = ReflectionResult(
            coverage_gaps=["gap"],
            contradictions=[],
            overlooked_angles=[],
            needs_more_research=True,
        )
        base_state["iteration"] = 0
        assert agents.should_continue(base_state) == "retriever"

    def test_should_continue_finishes_when_done(self, base_state):
        from core import agents
        from core.schemas import ReflectionResult

        base_state["reflection"] = ReflectionResult(
            coverage_gaps=[],
            contradictions=[],
            overlooked_angles=[],
            needs_more_research=False,
        )
        assert agents.should_continue(base_state) == "visualization"
