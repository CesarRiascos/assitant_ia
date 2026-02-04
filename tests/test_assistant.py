"""Tests for the Academic Assistant module."""

from unittest.mock import MagicMock, patch

import pytest

from academic_assistant.assistant import (
    AcademicAssistant,
    AssistantResponse,
    ConversationTurn,
)
from academic_assistant.research_apis import Paper


class TestConversationTurn:
    """Tests for ConversationTurn model."""

    def test_creation(self):
        """Test creating a conversation turn."""
        turn = ConversationTurn(role="user", content="Hello")
        assert turn.role == "user"
        assert turn.content == "Hello"
        assert turn.metadata == {}

    def test_with_metadata(self):
        """Test turn with metadata."""
        turn = ConversationTurn(
            role="assistant",
            content="Response",
            metadata={"tokens": 100},
        )
        assert turn.metadata["tokens"] == 100


class TestAssistantResponse:
    """Tests for AssistantResponse model."""

    def test_default_values(self):
        """Test default response values."""
        response = AssistantResponse(message="Test message")
        assert response.message == "Test message"
        assert response.papers_found == []
        assert response.documents_referenced == []
        assert response.suggestions == []

    def test_with_papers(self):
        """Test response with papers."""
        papers = [Paper(title="Test Paper")]
        response = AssistantResponse(
            message="Found papers",
            papers_found=papers,
            suggestions=["Try another search"],
        )
        assert len(response.papers_found) == 1
        assert len(response.suggestions) == 1


class TestAcademicAssistant:
    """Tests for AcademicAssistant class."""

    @patch("academic_assistant.assistant.ChatOpenAI")
    @patch("academic_assistant.assistant.OpenAIEmbeddings")
    def test_initialization(self, mock_embeddings, mock_llm):
        """Test assistant initialization."""
        assistant = AcademicAssistant(
            model_name="gpt-4",
            temperature=0.5,
            api_key="test_key",
        )
        assert assistant.model_name == "gpt-4"
        assert assistant.temperature == 0.5
        assert assistant.api_key == "test_key"

    @patch("academic_assistant.assistant.ChatOpenAI")
    @patch("academic_assistant.assistant.OpenAIEmbeddings")
    def test_clear_history(self, mock_embeddings, mock_llm):
        """Test clearing conversation history."""
        assistant = AcademicAssistant(api_key="test")
        assistant.conversation_history = [
            ConversationTurn(role="user", content="Test"),
            ConversationTurn(role="assistant", content="Response"),
        ]
        assistant.clear_history()
        assert len(assistant.conversation_history) == 0

    @patch("academic_assistant.assistant.ChatOpenAI")
    @patch("academic_assistant.assistant.OpenAIEmbeddings")
    def test_clear_documents(self, mock_embeddings, mock_llm):
        """Test clearing loaded documents."""
        assistant = AcademicAssistant(api_key="test")
        assistant.loaded_documents = {"test": MagicMock()}
        assistant.vector_store = MagicMock()
        assistant.clear_documents()
        assert len(assistant.loaded_documents) == 0
        assert assistant.vector_store is None

    @patch("academic_assistant.assistant.ChatOpenAI")
    @patch("academic_assistant.assistant.OpenAIEmbeddings")
    def test_get_loaded_documents_info_empty(self, mock_embeddings, mock_llm):
        """Test getting info when no documents loaded."""
        assistant = AcademicAssistant(api_key="test")
        info = assistant.get_loaded_documents_info()
        assert info == []

    @patch("academic_assistant.assistant.ChatOpenAI")
    @patch("academic_assistant.assistant.OpenAIEmbeddings")
    @patch.object(AcademicAssistant, "search_papers")
    def test_chat_without_search(self, mock_search, mock_embeddings, mock_llm):
        """Test chat without paper search."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = MagicMock(content="Test response")
        mock_llm.return_value = mock_llm_instance

        assistant = AcademicAssistant(api_key="test")
        response = assistant.chat("Hello", search_papers=False)

        assert response.message == "Test response"
        mock_search.assert_not_called()

    @patch("academic_assistant.assistant.ChatOpenAI")
    @patch("academic_assistant.assistant.OpenAIEmbeddings")
    @patch.object(AcademicAssistant, "search_papers")
    def test_chat_with_search(self, mock_search, mock_embeddings, mock_llm):
        """Test chat with paper search."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = MagicMock(content="Found relevant papers")
        mock_llm.return_value = mock_llm_instance

        mock_search.return_value = [
            Paper(title="Relevant Paper", citation_count=100)
        ]

        assistant = AcademicAssistant(api_key="test")
        response = assistant.chat("Find papers about AI", search_papers=True)

        assert "Found relevant papers" in response.message
        assert len(response.papers_found) == 1
        mock_search.assert_called_once()

    @patch("academic_assistant.assistant.ChatOpenAI")
    @patch("academic_assistant.assistant.OpenAIEmbeddings")
    def test_search_papers(self, mock_embeddings, mock_llm):
        """Test paper search functionality."""
        assistant = AcademicAssistant(api_key="test")

        # Mock the searcher
        mock_papers = [Paper(title="Test Paper", source="semantic_scholar")]
        assistant.searcher.search_all_and_deduplicate = MagicMock(return_value=mock_papers)

        papers = assistant.search_papers("machine learning", max_results=5)
        
        assert len(papers) == 1
        assert papers[0].title == "Test Paper"

    @patch("academic_assistant.assistant.ChatOpenAI")
    @patch("academic_assistant.assistant.OpenAIEmbeddings")
    def test_conversation_history_updated(self, mock_embeddings, mock_llm):
        """Test that conversation history is updated after chat."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = MagicMock(content="Response")
        mock_llm.return_value = mock_llm_instance

        assistant = AcademicAssistant(api_key="test")
        assistant.chat("Test message", search_papers=False)

        assert len(assistant.conversation_history) == 2
        assert assistant.conversation_history[0].role == "user"
        assert assistant.conversation_history[0].content == "Test message"
        assert assistant.conversation_history[1].role == "assistant"

    @patch("academic_assistant.assistant.ChatOpenAI")
    @patch("academic_assistant.assistant.OpenAIEmbeddings")
    def test_compare_papers_insufficient(self, mock_embeddings, mock_llm):
        """Test comparing fewer than 2 papers."""
        assistant = AcademicAssistant(api_key="test")
        result = assistant.compare_papers([Paper(title="Only one")])
        assert "at least 2 papers" in result

    @patch("academic_assistant.assistant.ChatOpenAI")
    @patch("academic_assistant.assistant.OpenAIEmbeddings")
    def test_generate_suggestions(self, mock_embeddings, mock_llm):
        """Test suggestion generation."""
        assistant = AcademicAssistant(api_key="test")
        
        papers = [Paper(title="Test Paper", citation_count=200)]
        suggestions = assistant._generate_suggestions("test query", papers)
        
        assert len(suggestions) > 0
        assert any("highly-cited" in s.lower() for s in suggestions)
