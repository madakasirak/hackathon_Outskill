"""Shared pytest fixtures.

Mocks the LLMs so tests run without API keys and without network calls.
This is the pattern: each test patches the global LLM objects in core.agents
to return predetermined Pydantic models.
"""
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_llm():
    """A reusable mock LLM. Use the helper methods to set return values per test."""
    llm = MagicMock()

    def with_structured_output(schema):
        # The real method returns an LLM bound to the schema; we just return self
        # and let the test set llm.invoke.return_value to a schema instance.
        return llm

    llm.with_structured_output = MagicMock(side_effect=with_structured_output)
    llm.invoke = MagicMock()
    return llm


@pytest.fixture
def fake_vector_store():
    """A mock ChromaDB. similarity_search returns a fixed list of Documents."""
    from langchain_core.documents import Document

    store = MagicMock()
    store.similarity_search = MagicMock(return_value=[
        Document(
            page_content="Mocked evidence chunk for fact-checking.",
            metadata={"source": "tavily", "url": "https://example.com"},
        )
    ])
    store.add_documents = MagicMock(return_value=None)
    return store


@pytest.fixture
def base_state():
    """A minimal valid ResearchState for test inputs."""
    return {
        "query": "Test research query",
        "raw_findings": [],
        "documents": [],
        "fact_checks": [],
        "iteration": 0,
        "source_assessments": [],
    }


@pytest.fixture
def sample_documents():
    """A few canned Documents to seed retriever output tests."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="Government report on EV adoption shows 8% growth in 2024.",
            metadata={"source": "tavily", "url": "https://gov.example/ev-report"},
        ),
        Document(
            page_content="Battery cost analysis: $137/kWh average in 2024, down 14% YoY.",
            metadata={"source": "arxiv", "url": "https://arxiv.org/abs/0000.00000"},
        ),
        Document(
            page_content="Charging infrastructure remains a bottleneck in rural areas.",
            metadata={"source": "wikipedia", "url": "https://en.wikipedia.org/wiki/EV"},
        ),
    ]
