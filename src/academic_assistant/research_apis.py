"""Academic Research APIs Module.

This module provides clients for various academic research APIs:
- Semantic Scholar
- arXiv
- CrossRef
- OpenAlex

All APIs use public endpoints with optional API keys for higher rate limits.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

import aiohttp
import arxiv
import requests
from pydantic import BaseModel, Field

from academic_assistant.config import settings


class Paper(BaseModel):
    """Unified paper representation across all APIs."""

    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    citation_count: Optional[int] = None
    venue: Optional[str] = None
    source: str = "unknown"
    external_ids: dict[str, str] = Field(default_factory=dict)

    def to_citation(self) -> str:
        """Generate a citation string for the paper."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."

        year_str = f" ({self.year})" if self.year else ""
        venue_str = f". {self.venue}" if self.venue else ""
        doi_str = f". DOI: {self.doi}" if self.doi else ""

        return f"{authors_str}{year_str}. {self.title}{venue_str}{doi_str}"


class SearchResult(BaseModel):
    """Search result from an academic API."""

    papers: list[Paper] = Field(default_factory=list)
    total_results: Optional[int] = None
    query: str = ""
    source: str = "unknown"
    search_time: Optional[float] = None


class ResearchAPIClient(ABC):
    """Abstract base class for research API clients."""

    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> SearchResult:
        """Search for papers matching the query."""
        pass

    @abstractmethod
    async def async_search(self, query: str, max_results: int = 10) -> SearchResult:
        """Async search for papers matching the query."""
        pass

    @abstractmethod
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get a specific paper by ID."""
        pass


class SemanticScholarClient(ResearchAPIClient):
    """
    Client for the Semantic Scholar API.

    Semantic Scholar provides access to a large corpus of academic papers
    with citation data and semantic features.

    API Documentation: https://api.semanticscholar.org/
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    SEARCH_FIELDS = "title,authors,abstract,year,venue,citationCount,externalIds,url,openAccessPdf"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Semantic Scholar client.

        Args:
            api_key: Optional API key for higher rate limits.
        """
        self.api_key = api_key or settings.api.semantic_scholar_api_key
        self.headers = {}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key

    def _parse_paper(self, data: dict) -> Paper:
        """Parse a paper from Semantic Scholar response."""
        authors = [a.get("name", "") for a in data.get("authors", [])]
        external_ids = data.get("externalIds", {}) or {}

        pdf_url = None
        open_access = data.get("openAccessPdf")
        if open_access and isinstance(open_access, dict):
            pdf_url = open_access.get("url")

        return Paper(
            title=data.get("title", ""),
            authors=authors,
            abstract=data.get("abstract"),
            year=data.get("year"),
            doi=external_ids.get("DOI"),
            url=data.get("url"),
            pdf_url=pdf_url,
            citation_count=data.get("citationCount"),
            venue=data.get("venue"),
            source="semantic_scholar",
            external_ids=external_ids,
        )

    def search(self, query: str, max_results: int = 10) -> SearchResult:
        """
        Search for papers on Semantic Scholar.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            SearchResult with matching papers.
        """
        start_time = datetime.now()
        max_results = min(max_results, settings.search.max_results)

        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": self.SEARCH_FIELDS,
        }

        try:
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                timeout=settings.search.timeout,
            )
            response.raise_for_status()
            data = response.json()

            papers = [self._parse_paper(p) for p in data.get("data", [])]
            total = data.get("total", len(papers))
            search_time = (datetime.now() - start_time).total_seconds()

            return SearchResult(
                papers=papers,
                total_results=total,
                query=query,
                source="semantic_scholar",
                search_time=search_time,
            )
        except requests.RequestException as e:
            return SearchResult(
                papers=[],
                query=query,
                source="semantic_scholar",
                search_time=(datetime.now() - start_time).total_seconds(),
            )

    async def async_search(self, query: str, max_results: int = 10) -> SearchResult:
        """Async search for papers on Semantic Scholar."""
        start_time = datetime.now()
        max_results = min(max_results, settings.search.max_results)

        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": self.SEARCH_FIELDS,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=settings.search.timeout),
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

                    papers = [self._parse_paper(p) for p in data.get("data", [])]
                    total = data.get("total", len(papers))
                    search_time = (datetime.now() - start_time).total_seconds()

                    return SearchResult(
                        papers=papers,
                        total_results=total,
                        query=query,
                        source="semantic_scholar",
                        search_time=search_time,
                    )
        except Exception:
            return SearchResult(
                papers=[],
                query=query,
                source="semantic_scholar",
                search_time=(datetime.now() - start_time).total_seconds(),
            )

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get a specific paper by Semantic Scholar ID."""
        url = f"{self.BASE_URL}/paper/{paper_id}"
        params = {"fields": self.SEARCH_FIELDS}

        try:
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                timeout=settings.search.timeout,
            )
            response.raise_for_status()
            return self._parse_paper(response.json())
        except requests.RequestException:
            return None


class ArxivClient(ResearchAPIClient):
    """
    Client for the arXiv API.

    arXiv is a free distribution service and open archive for scholarly articles.

    API Documentation: https://arxiv.org/help/api
    """

    def __init__(self):
        """Initialize arXiv client."""
        self.client = arxiv.Client()

    def _parse_paper(self, result: arxiv.Result) -> Paper:
        """Parse a paper from arXiv result."""
        return Paper(
            title=result.title,
            authors=[a.name for a in result.authors],
            abstract=result.summary,
            year=result.published.year if result.published else None,
            doi=result.doi,
            url=result.entry_id,
            pdf_url=result.pdf_url,
            citation_count=None,  # arXiv doesn't provide citation counts
            venue="arXiv",
            source="arxiv",
            external_ids={"arxiv": result.entry_id.split("/")[-1]},
        )

    def search(self, query: str, max_results: int = 10) -> SearchResult:
        """
        Search for papers on arXiv.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            SearchResult with matching papers.
        """
        start_time = datetime.now()
        max_results = min(max_results, settings.search.max_results)

        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            papers = [self._parse_paper(r) for r in self.client.results(search)]
            search_time = (datetime.now() - start_time).total_seconds()

            return SearchResult(
                papers=papers,
                total_results=len(papers),
                query=query,
                source="arxiv",
                search_time=search_time,
            )
        except Exception:
            return SearchResult(
                papers=[],
                query=query,
                source="arxiv",
                search_time=(datetime.now() - start_time).total_seconds(),
            )

    async def async_search(self, query: str, max_results: int = 10) -> SearchResult:
        """Async search for papers on arXiv (runs sync in executor)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search, query, max_results)

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get a specific paper by arXiv ID."""
        try:
            search = arxiv.Search(id_list=[paper_id])
            results = list(self.client.results(search))
            if results:
                return self._parse_paper(results[0])
            return None
        except Exception:
            return None


class CrossRefClient(ResearchAPIClient):
    """
    Client for the CrossRef API.

    CrossRef provides DOI registration and metadata for scholarly content.

    API Documentation: https://api.crossref.org/
    """

    BASE_URL = "https://api.crossref.org"

    def __init__(self, email: Optional[str] = None):
        """
        Initialize CrossRef client.

        Args:
            email: Email for polite pool (higher rate limits).
        """
        self.email = email or settings.api.crossref_email
        self.headers = {"User-Agent": f"AcademicAssistant/0.1.0"}
        if self.email:
            self.headers["User-Agent"] += f" (mailto:{self.email})"

    def _parse_paper(self, data: dict) -> Paper:
        """Parse a paper from CrossRef response."""
        # Extract authors
        authors = []
        for author in data.get("author", []):
            given = author.get("given", "")
            family = author.get("family", "")
            if given and family:
                authors.append(f"{given} {family}")
            elif family:
                authors.append(family)

        # Extract title
        title_list = data.get("title", [])
        title = title_list[0] if title_list else ""

        # Extract abstract
        abstract = data.get("abstract", "")
        if abstract:
            # Clean HTML from abstract
            from bs4 import BeautifulSoup
            abstract = BeautifulSoup(abstract, "html.parser").get_text()

        # Extract year
        year = None
        date_parts = data.get("published-print", {}).get("date-parts", [[]])
        if not date_parts or not date_parts[0]:
            date_parts = data.get("published-online", {}).get("date-parts", [[]])
        if date_parts and date_parts[0]:
            year = date_parts[0][0]

        # Extract venue
        venue = None
        container = data.get("container-title", [])
        if container:
            venue = container[0]

        doi = data.get("DOI")
        url = f"https://doi.org/{doi}" if doi else None

        return Paper(
            title=title,
            authors=authors,
            abstract=abstract if abstract else None,
            year=year,
            doi=doi,
            url=url,
            pdf_url=None,
            citation_count=data.get("is-referenced-by-count"),
            venue=venue,
            source="crossref",
            external_ids={"DOI": doi} if doi else {},
        )

    def search(self, query: str, max_results: int = 10) -> SearchResult:
        """
        Search for papers on CrossRef.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            SearchResult with matching papers.
        """
        start_time = datetime.now()
        max_results = min(max_results, settings.search.max_results)

        url = f"{self.BASE_URL}/works"
        params = {
            "query": query,
            "rows": max_results,
        }

        try:
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                timeout=settings.search.timeout,
            )
            response.raise_for_status()
            data = response.json()

            items = data.get("message", {}).get("items", [])
            papers = [self._parse_paper(item) for item in items]
            total = data.get("message", {}).get("total-results", len(papers))
            search_time = (datetime.now() - start_time).total_seconds()

            return SearchResult(
                papers=papers,
                total_results=total,
                query=query,
                source="crossref",
                search_time=search_time,
            )
        except requests.RequestException:
            return SearchResult(
                papers=[],
                query=query,
                source="crossref",
                search_time=(datetime.now() - start_time).total_seconds(),
            )

    async def async_search(self, query: str, max_results: int = 10) -> SearchResult:
        """Async search for papers on CrossRef."""
        start_time = datetime.now()
        max_results = min(max_results, settings.search.max_results)

        url = f"{self.BASE_URL}/works"
        params = {
            "query": query,
            "rows": max_results,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=settings.search.timeout),
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

                    items = data.get("message", {}).get("items", [])
                    papers = [self._parse_paper(item) for item in items]
                    total = data.get("message", {}).get("total-results", len(papers))
                    search_time = (datetime.now() - start_time).total_seconds()

                    return SearchResult(
                        papers=papers,
                        total_results=total,
                        query=query,
                        source="crossref",
                        search_time=search_time,
                    )
        except Exception:
            return SearchResult(
                papers=[],
                query=query,
                source="crossref",
                search_time=(datetime.now() - start_time).total_seconds(),
            )

    def get_paper(self, doi: str) -> Optional[Paper]:
        """Get a specific paper by DOI."""
        url = f"{self.BASE_URL}/works/{doi}"

        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=settings.search.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_paper(data.get("message", {}))
        except requests.RequestException:
            return None


class OpenAlexClient(ResearchAPIClient):
    """
    Client for the OpenAlex API.

    OpenAlex is an open catalog of scholarly works, authors, venues, and institutions.

    API Documentation: https://docs.openalex.org/
    """

    BASE_URL = "https://api.openalex.org"

    def __init__(self, email: Optional[str] = None):
        """
        Initialize OpenAlex client.

        Args:
            email: Email for polite pool (faster responses).
        """
        self.email = email or settings.api.crossref_email
        self.params = {}
        if self.email:
            self.params["mailto"] = self.email

    def _parse_paper(self, data: dict) -> Paper:
        """Parse a paper from OpenAlex response."""
        # Extract authors
        authors = []
        for authorship in data.get("authorships", []):
            author = authorship.get("author", {})
            name = author.get("display_name")
            if name:
                authors.append(name)

        # Extract title
        title = data.get("title", "")

        # Extract abstract
        abstract = None
        abstract_inverted = data.get("abstract_inverted_index")
        if abstract_inverted:
            # Reconstruct abstract from inverted index
            word_positions = []
            for word, positions in abstract_inverted.items():
                for pos in positions:
                    word_positions.append((pos, word))
            word_positions.sort()
            abstract = " ".join(word for _, word in word_positions)

        # Extract year
        year = data.get("publication_year")

        # Extract DOI
        doi = data.get("doi")
        if doi and doi.startswith("https://doi.org/"):
            doi = doi[16:]

        # Extract venue
        venue = None
        location = data.get("primary_location", {})
        if location:
            source = location.get("source", {})
            if source:
                venue = source.get("display_name")

        # Extract PDF URL
        pdf_url = None
        best_oa = data.get("best_oa_location", {})
        if best_oa:
            pdf_url = best_oa.get("pdf_url")

        return Paper(
            title=title,
            authors=authors,
            abstract=abstract,
            year=year,
            doi=doi,
            url=data.get("id"),
            pdf_url=pdf_url,
            citation_count=data.get("cited_by_count"),
            venue=venue,
            source="openalex",
            external_ids=data.get("ids", {}),
        )

    def search(self, query: str, max_results: int = 10) -> SearchResult:
        """
        Search for papers on OpenAlex.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            SearchResult with matching papers.
        """
        start_time = datetime.now()
        max_results = min(max_results, settings.search.max_results)

        url = f"{self.BASE_URL}/works"
        params = {
            **self.params,
            "search": query,
            "per_page": max_results,
        }

        try:
            response = requests.get(
                url,
                params=params,
                timeout=settings.search.timeout,
            )
            response.raise_for_status()
            data = response.json()

            papers = [self._parse_paper(item) for item in data.get("results", [])]
            total = data.get("meta", {}).get("count", len(papers))
            search_time = (datetime.now() - start_time).total_seconds()

            return SearchResult(
                papers=papers,
                total_results=total,
                query=query,
                source="openalex",
                search_time=search_time,
            )
        except requests.RequestException:
            return SearchResult(
                papers=[],
                query=query,
                source="openalex",
                search_time=(datetime.now() - start_time).total_seconds(),
            )

    async def async_search(self, query: str, max_results: int = 10) -> SearchResult:
        """Async search for papers on OpenAlex."""
        start_time = datetime.now()
        max_results = min(max_results, settings.search.max_results)

        url = f"{self.BASE_URL}/works"
        params = {
            **self.params,
            "search": query,
            "per_page": max_results,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=settings.search.timeout),
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

                    papers = [self._parse_paper(item) for item in data.get("results", [])]
                    total = data.get("meta", {}).get("count", len(papers))
                    search_time = (datetime.now() - start_time).total_seconds()

                    return SearchResult(
                        papers=papers,
                        total_results=total,
                        query=query,
                        source="openalex",
                        search_time=search_time,
                    )
        except Exception:
            return SearchResult(
                papers=[],
                query=query,
                source="openalex",
                search_time=(datetime.now() - start_time).total_seconds(),
            )

    def get_paper(self, work_id: str) -> Optional[Paper]:
        """Get a specific work by OpenAlex ID."""
        url = f"{self.BASE_URL}/works/{work_id}"
        params = self.params.copy()

        try:
            response = requests.get(
                url,
                params=params,
                timeout=settings.search.timeout,
            )
            response.raise_for_status()
            return self._parse_paper(response.json())
        except requests.RequestException:
            return None


class MultiSourceSearcher:
    """
    Search across multiple academic sources simultaneously.

    Aggregates results from Semantic Scholar, arXiv, CrossRef, and OpenAlex.
    """

    def __init__(self):
        """Initialize multi-source searcher with all API clients."""
        self.clients: dict[str, ResearchAPIClient] = {
            "semantic_scholar": SemanticScholarClient(),
            "arxiv": ArxivClient(),
            "crossref": CrossRefClient(),
            "openalex": OpenAlexClient(),
        }

    def search(
        self,
        query: str,
        sources: Optional[list[str]] = None,
        max_results_per_source: int = 5,
    ) -> dict[str, SearchResult]:
        """
        Search across multiple sources.

        Args:
            query: Search query string.
            sources: List of sources to search. None means all sources.
            max_results_per_source: Maximum results from each source.

        Returns:
            Dictionary mapping source name to SearchResult.
        """
        if sources is None:
            sources = list(self.clients.keys())

        results = {}
        for source in sources:
            if source in self.clients:
                results[source] = self.clients[source].search(query, max_results_per_source)

        return results

    async def async_search(
        self,
        query: str,
        sources: Optional[list[str]] = None,
        max_results_per_source: int = 5,
    ) -> dict[str, SearchResult]:
        """
        Async search across multiple sources.

        Args:
            query: Search query string.
            sources: List of sources to search. None means all sources.
            max_results_per_source: Maximum results from each source.

        Returns:
            Dictionary mapping source name to SearchResult.
        """
        if sources is None:
            sources = list(self.clients.keys())

        tasks = []
        for source in sources:
            if source in self.clients:
                tasks.append(
                    (source, self.clients[source].async_search(query, max_results_per_source))
                )

        results = {}
        for source, task in tasks:
            results[source] = await task

        return results

    def search_all_and_deduplicate(
        self,
        query: str,
        max_results: int = 20,
    ) -> list[Paper]:
        """
        Search all sources and deduplicate results by DOI.

        Args:
            query: Search query string.
            max_results: Maximum total results to return.

        Returns:
            Deduplicated list of papers.
        """
        all_results = self.search(query, max_results_per_source=10)

        # Collect all papers
        all_papers = []
        for result in all_results.values():
            all_papers.extend(result.papers)

        # Deduplicate by DOI
        seen_dois: set[str] = set()
        seen_titles: set[str] = set()
        unique_papers = []

        for paper in all_papers:
            # Check by DOI
            if paper.doi:
                if paper.doi in seen_dois:
                    continue
                seen_dois.add(paper.doi)

            # Check by normalized title
            normalized_title = paper.title.lower().strip()
            if normalized_title in seen_titles:
                continue
            seen_titles.add(normalized_title)

            unique_papers.append(paper)

        # Sort by citation count (descending)
        unique_papers.sort(
            key=lambda p: p.citation_count or 0,
            reverse=True,
        )

        return unique_papers[:max_results]
