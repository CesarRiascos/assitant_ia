"""Tests for the Research APIs module."""

from unittest.mock import MagicMock, patch

import pytest

from academic_assistant.research_apis import (
    Paper,
    SearchResult,
    SemanticScholarClient,
    ArxivClient,
    CrossRefClient,
    OpenAlexClient,
    MultiSourceSearcher,
)


class TestPaper:
    """Tests for Paper model."""

    def test_default_values(self):
        """Test default values for paper."""
        paper = Paper(title="Test Paper")
        assert paper.title == "Test Paper"
        assert paper.authors == []
        assert paper.abstract is None
        assert paper.source == "unknown"

    def test_with_values(self):
        """Test paper with all values."""
        paper = Paper(
            title="Test Paper",
            authors=["John Doe", "Jane Smith"],
            abstract="This is an abstract.",
            year=2024,
            doi="10.1234/test",
            citation_count=100,
            source="semantic_scholar",
        )
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert paper.year == 2024
        assert paper.doi == "10.1234/test"

    def test_to_citation(self):
        """Test citation generation."""
        paper = Paper(
            title="Test Paper",
            authors=["John Doe", "Jane Smith"],
            year=2024,
            doi="10.1234/test",
            venue="Nature",
        )
        citation = paper.to_citation()
        assert "John Doe" in citation
        assert "Jane Smith" in citation
        assert "2024" in citation
        assert "Test Paper" in citation
        assert "10.1234/test" in citation

    def test_to_citation_many_authors(self):
        """Test citation with many authors."""
        paper = Paper(
            title="Test Paper",
            authors=["Author 1", "Author 2", "Author 3", "Author 4"],
            year=2024,
        )
        citation = paper.to_citation()
        assert "et al." in citation


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_default_values(self):
        """Test default values for search result."""
        result = SearchResult()
        assert result.papers == []
        assert result.total_results is None
        assert result.query == ""
        assert result.source == "unknown"

    def test_with_papers(self):
        """Test search result with papers."""
        papers = [
            Paper(title="Paper 1"),
            Paper(title="Paper 2"),
        ]
        result = SearchResult(
            papers=papers,
            total_results=100,
            query="test query",
            source="semantic_scholar",
        )
        assert len(result.papers) == 2
        assert result.total_results == 100


class TestSemanticScholarClient:
    """Tests for SemanticScholarClient."""

    def test_initialization(self):
        """Test client initialization."""
        client = SemanticScholarClient()
        assert client.BASE_URL == "https://api.semanticscholar.org/graph/v1"

    def test_initialization_with_api_key(self):
        """Test client initialization with API key."""
        client = SemanticScholarClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert "x-api-key" in client.headers

    @patch("academic_assistant.research_apis.requests.get")
    def test_search_success(self, mock_get):
        """Test successful search."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "title": "Test Paper",
                    "authors": [{"name": "John Doe"}],
                    "year": 2024,
                    "citationCount": 50,
                }
            ],
            "total": 1,
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = SemanticScholarClient()
        result = client.search("machine learning", max_results=5)

        assert len(result.papers) == 1
        assert result.papers[0].title == "Test Paper"
        assert result.source == "semantic_scholar"

    @patch("academic_assistant.research_apis.requests.get")
    def test_search_failure(self, mock_get):
        """Test search failure handling."""
        mock_get.side_effect = Exception("Network error")

        client = SemanticScholarClient()
        result = client.search("test query")

        assert len(result.papers) == 0
        assert result.source == "semantic_scholar"


class TestArxivClient:
    """Tests for ArxivClient."""

    def test_initialization(self):
        """Test client initialization."""
        client = ArxivClient()
        assert client.client is not None

    @patch("academic_assistant.research_apis.arxiv.Client")
    def test_search_success(self, mock_client_class):
        """Test successful search."""
        mock_result = MagicMock()
        mock_result.title = "Test Paper"
        mock_result.authors = [MagicMock(name="John Doe")]
        mock_result.summary = "Abstract text"
        mock_result.published = MagicMock(year=2024)
        mock_result.doi = "10.1234/test"
        mock_result.entry_id = "http://arxiv.org/abs/2024.12345"
        mock_result.pdf_url = "http://arxiv.org/pdf/2024.12345"

        mock_client = MagicMock()
        mock_client.results.return_value = [mock_result]
        mock_client_class.return_value = mock_client

        client = ArxivClient()
        client.client = mock_client
        result = client.search("machine learning")

        assert result.source == "arxiv"


class TestCrossRefClient:
    """Tests for CrossRefClient."""

    def test_initialization(self):
        """Test client initialization."""
        client = CrossRefClient()
        assert client.BASE_URL == "https://api.crossref.org"

    def test_initialization_with_email(self):
        """Test client initialization with email."""
        client = CrossRefClient(email="test@example.com")
        assert "test@example.com" in client.headers["User-Agent"]

    @patch("academic_assistant.research_apis.requests.get")
    def test_search_success(self, mock_get):
        """Test successful search."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "items": [
                    {
                        "title": ["Test Paper"],
                        "author": [{"given": "John", "family": "Doe"}],
                        "published-print": {"date-parts": [[2024]]},
                        "DOI": "10.1234/test",
                        "is-referenced-by-count": 50,
                    }
                ],
                "total-results": 1,
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = CrossRefClient()
        result = client.search("test query")

        assert len(result.papers) == 1
        assert result.source == "crossref"


class TestOpenAlexClient:
    """Tests for OpenAlexClient."""

    def test_initialization(self):
        """Test client initialization."""
        client = OpenAlexClient()
        assert client.BASE_URL == "https://api.openalex.org"

    @patch("academic_assistant.research_apis.requests.get")
    def test_search_success(self, mock_get):
        """Test successful search."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Test Paper",
                    "authorships": [{"author": {"display_name": "John Doe"}}],
                    "publication_year": 2024,
                    "doi": "https://doi.org/10.1234/test",
                    "cited_by_count": 50,
                }
            ],
            "meta": {"count": 1},
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = OpenAlexClient()
        result = client.search("test query")

        assert len(result.papers) == 1
        assert result.source == "openalex"


class TestMultiSourceSearcher:
    """Tests for MultiSourceSearcher."""

    def test_initialization(self):
        """Test searcher initialization."""
        searcher = MultiSourceSearcher()
        assert "semantic_scholar" in searcher.clients
        assert "arxiv" in searcher.clients
        assert "crossref" in searcher.clients
        assert "openalex" in searcher.clients

    @patch.object(SemanticScholarClient, "search")
    @patch.object(ArxivClient, "search")
    def test_search_specific_sources(self, mock_arxiv, mock_ss):
        """Test searching specific sources."""
        mock_ss.return_value = SearchResult(
            papers=[Paper(title="SS Paper")],
            source="semantic_scholar",
        )
        mock_arxiv.return_value = SearchResult(
            papers=[Paper(title="arXiv Paper")],
            source="arxiv",
        )

        searcher = MultiSourceSearcher()
        results = searcher.search("test", sources=["semantic_scholar", "arxiv"])

        assert "semantic_scholar" in results
        assert "arxiv" in results
        assert "crossref" not in results

    @patch.object(SemanticScholarClient, "search")
    @patch.object(ArxivClient, "search")
    @patch.object(CrossRefClient, "search")
    @patch.object(OpenAlexClient, "search")
    def test_search_all_and_deduplicate(self, mock_oa, mock_cr, mock_arxiv, mock_ss):
        """Test searching all sources with deduplication."""
        # Create papers with same DOI
        paper1 = Paper(title="Same Paper", doi="10.1234/test", source="semantic_scholar")
        paper2 = Paper(title="Same Paper", doi="10.1234/test", source="crossref")
        paper3 = Paper(title="Different Paper", doi="10.5678/other", source="arxiv")

        mock_ss.return_value = SearchResult(papers=[paper1], source="semantic_scholar")
        mock_arxiv.return_value = SearchResult(papers=[paper3], source="arxiv")
        mock_cr.return_value = SearchResult(papers=[paper2], source="crossref")
        mock_oa.return_value = SearchResult(papers=[], source="openalex")

        searcher = MultiSourceSearcher()
        papers = searcher.search_all_and_deduplicate("test", max_results=10)

        # Should deduplicate by DOI
        assert len(papers) == 2
