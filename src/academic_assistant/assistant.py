"""Academic Research Assistant Module.

This module provides the main AI assistant for academic investigations,
integrating PDF processing, research APIs, and LLM capabilities.
"""

import asyncio
from pathlib import Path
from typing import Any, Optional, Union

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from academic_assistant.config import settings
from academic_assistant.pdf_loader import PDFDocument, PDFLoader
from academic_assistant.research_apis import (
    MultiSourceSearcher,
    Paper,
    SearchResult,
)


SYSTEM_PROMPT = """You are an AI research assistant specialized in academic investigations. 
Your role is to help researchers with:

1. **Literature Review**: Search and analyze academic papers from multiple sources
2. **Document Analysis**: Extract insights from PDF documents
3. **Research Synthesis**: Summarize and synthesize findings across multiple papers
4. **Citation Generation**: Help format citations and references
5. **Research Questions**: Help formulate and refine research questions

When searching for papers, you have access to these academic databases:
- Semantic Scholar: Comprehensive academic paper database with citation data
- arXiv: Preprint server for physics, mathematics, computer science, and more
- CrossRef: DOI registration agency with metadata for scholarly content
- OpenAlex: Open catalog of scholarly works

Guidelines:
- Provide accurate, well-sourced information
- Cite papers when making claims
- Acknowledge limitations in available data
- Suggest relevant follow-up queries
- Be precise about what you know vs. what you're uncertain about

Current context:
{context}
"""


class ConversationTurn(BaseModel):
    """A single turn in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    metadata: dict = Field(default_factory=dict)


class AssistantResponse(BaseModel):
    """Response from the academic assistant."""

    message: str
    papers_found: list[Paper] = Field(default_factory=list)
    documents_referenced: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    sources_searched: list[str] = Field(default_factory=list)


class AcademicAssistant:
    """
    AI-powered Academic Research Assistant.

    Combines PDF processing, academic API search, and LLM capabilities
    to assist with academic investigations.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Academic Assistant.

        Args:
            model_name: Name of the OpenAI model to use.
            temperature: Temperature for response generation.
            api_key: OpenAI API key. If not provided, uses environment variable.
        """
        self.model_name = model_name or settings.model.model_name
        self.temperature = temperature if temperature is not None else settings.model.temperature
        self.api_key = api_key or settings.api.openai_api_key

        # Initialize components
        self.pdf_loader = PDFLoader()
        self.searcher = MultiSourceSearcher()

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key,
        )

        # Initialize embeddings for document retrieval
        self.embeddings = OpenAIEmbeddings(api_key=self.api_key)

        # Conversation history
        self.conversation_history: list[ConversationTurn] = []

        # Loaded documents
        self.loaded_documents: dict[str, PDFDocument] = {}
        self.vector_store: Optional[FAISS] = None

        # Context for the assistant
        self.context = "No documents loaded. Ready to search academic databases."

    def _update_context(self) -> None:
        """Update the context based on loaded documents."""
        if not self.loaded_documents:
            self.context = "No documents loaded. Ready to search academic databases."
            return

        doc_summaries = []
        for name, doc in self.loaded_documents.items():
            summary = f"- {name}: {doc.metadata.title or 'Untitled'}"
            if doc.metadata.author:
                summary += f" by {doc.metadata.author}"
            summary += f" ({doc.metadata.num_pages} pages)"
            doc_summaries.append(summary)

        self.context = f"Loaded documents:\n" + "\n".join(doc_summaries)

    def load_pdf(self, file_path: Union[str, Path], name: Optional[str] = None) -> PDFDocument:
        """
        Load a PDF document into the assistant's context.

        Args:
            file_path: Path to the PDF file.
            name: Optional name for the document. Defaults to filename.

        Returns:
            Loaded PDFDocument.
        """
        file_path = Path(file_path)
        doc = self.pdf_loader.load_from_path(file_path)

        doc_name = name or file_path.stem
        self.loaded_documents[doc_name] = doc

        # Update vector store
        self._update_vector_store()
        self._update_context()

        return doc

    def load_pdf_bytes(self, data: bytes, name: str) -> PDFDocument:
        """
        Load a PDF document from bytes.

        Args:
            data: PDF content as bytes.
            name: Name for the document.

        Returns:
            Loaded PDFDocument.
        """
        doc = self.pdf_loader.load_from_bytes(data, name)
        self.loaded_documents[name] = doc

        self._update_vector_store()
        self._update_context()

        return doc

    def _update_vector_store(self) -> None:
        """Update the vector store with all loaded documents."""
        if not self.loaded_documents:
            self.vector_store = None
            return

        all_docs = []
        for name, pdf_doc in self.loaded_documents.items():
            for page in pdf_doc.pages:
                doc = Document(
                    page_content=page.text,
                    metadata={
                        "source": name,
                        "page": page.page_number,
                        "title": pdf_doc.metadata.title,
                    },
                )
                all_docs.append(doc)

        if all_docs:
            # Split documents into chunks
            chunks = self.pdf_loader.text_splitter.split_documents(all_docs)
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)

    def search_papers(
        self,
        query: str,
        sources: Optional[list[str]] = None,
        max_results: int = 10,
    ) -> list[Paper]:
        """
        Search for academic papers across multiple sources.

        Args:
            query: Search query string.
            sources: List of sources to search. None means all sources.
            max_results: Maximum number of results to return.

        Returns:
            List of found papers.
        """
        if sources:
            results = self.searcher.search(query, sources, max_results_per_source=max_results)
            papers = []
            for result in results.values():
                papers.extend(result.papers)
            return papers[:max_results]
        else:
            return self.searcher.search_all_and_deduplicate(query, max_results)

    async def async_search_papers(
        self,
        query: str,
        sources: Optional[list[str]] = None,
        max_results: int = 10,
    ) -> list[Paper]:
        """Async version of search_papers."""
        if sources:
            results = await self.searcher.async_search(query, sources, max_results_per_source=max_results)
            papers = []
            for result in results.values():
                papers.extend(result.papers)
            return papers[:max_results]
        else:
            # Run sync deduplication in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.searcher.search_all_and_deduplicate,
                query,
                max_results,
            )

    def _retrieve_relevant_context(self, query: str, k: int = 4) -> str:
        """Retrieve relevant context from loaded documents."""
        if not self.vector_store:
            return ""

        docs = self.vector_store.similarity_search(query, k=k)
        if not docs:
            return ""

        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            context_parts.append(f"[From {source}, page {page}]:\n{doc.page_content}")

        return "\n\n".join(context_parts)

    def chat(
        self,
        message: str,
        search_papers: bool = True,
        include_document_context: bool = True,
    ) -> AssistantResponse:
        """
        Send a message to the assistant and get a response.

        Args:
            message: User message.
            search_papers: Whether to search for relevant papers.
            include_document_context: Whether to include context from loaded documents.

        Returns:
            AssistantResponse with the assistant's reply.
        """
        # Build context
        document_context = ""
        if include_document_context and self.vector_store:
            document_context = self._retrieve_relevant_context(message)

        # Search for relevant papers if requested
        papers_found = []
        sources_searched = []
        if search_papers:
            # Extract search terms from message
            papers_found = self.search_papers(message, max_results=5)
            sources_searched = ["semantic_scholar", "arxiv", "crossref", "openalex"]

        # Build paper context
        paper_context = ""
        if papers_found:
            paper_summaries = []
            for i, paper in enumerate(papers_found[:5], 1):
                summary = f"{i}. {paper.title}"
                if paper.authors:
                    summary += f" ({', '.join(paper.authors[:2])}{'...' if len(paper.authors) > 2 else ''})"
                if paper.year:
                    summary += f" [{paper.year}]"
                if paper.citation_count:
                    summary += f" - {paper.citation_count} citations"
                paper_summaries.append(summary)
            paper_context = "\n\nRelevant papers found:\n" + "\n".join(paper_summaries)

        # Build full context
        full_context = self.context
        if document_context:
            full_context += f"\n\nRelevant excerpts from loaded documents:\n{document_context}"
        if paper_context:
            full_context += paper_context

        # Build messages
        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=full_context)),
        ]

        # Add conversation history
        for turn in self.conversation_history[-10:]:  # Keep last 10 turns
            if turn.role == "user":
                messages.append(HumanMessage(content=turn.content))
            else:
                messages.append(AIMessage(content=turn.content))

        # Add current message
        messages.append(HumanMessage(content=message))

        # Get response
        response = self.llm.invoke(messages)
        response_text = response.content

        # Update conversation history
        self.conversation_history.append(ConversationTurn(role="user", content=message))
        self.conversation_history.append(ConversationTurn(role="assistant", content=response_text))

        # Generate suggestions
        suggestions = self._generate_suggestions(message, papers_found)

        return AssistantResponse(
            message=response_text,
            papers_found=papers_found,
            documents_referenced=list(self.loaded_documents.keys()),
            suggestions=suggestions,
            sources_searched=sources_searched,
        )

    def _generate_suggestions(self, query: str, papers: list[Paper]) -> list[str]:
        """Generate follow-up suggestions based on the query and results."""
        suggestions = []

        if papers:
            # Suggest exploring specific papers
            top_paper = papers[0]
            if top_paper.citation_count and top_paper.citation_count > 100:
                suggestions.append(f"Explore highly-cited paper: '{top_paper.title}'")

            # Suggest related searches
            suggestions.append(f"Find papers that cite '{top_paper.title}'")

        if self.loaded_documents:
            suggestions.append("Summarize the key findings from loaded documents")

        suggestions.append("Refine search with more specific terms")

        return suggestions[:4]

    def summarize_document(self, doc_name: str) -> str:
        """
        Generate a summary of a loaded document.

        Args:
            doc_name: Name of the loaded document.

        Returns:
            Summary of the document.
        """
        if doc_name not in self.loaded_documents:
            return f"Document '{doc_name}' not found. Loaded documents: {list(self.loaded_documents.keys())}"

        doc = self.loaded_documents[doc_name]
        text = doc.get_text()

        # Truncate if too long
        max_chars = 15000
        if len(text) > max_chars:
            text = text[:max_chars] + "...[truncated]"

        prompt = f"""Please provide a comprehensive summary of the following academic document:

Title: {doc.metadata.title or 'Unknown'}
Author: {doc.metadata.author or 'Unknown'}
Pages: {doc.metadata.num_pages}

Content:
{text}

Provide a summary that includes:
1. Main research question or objective
2. Key methodology
3. Main findings/results
4. Conclusions and implications
"""

        messages = [
            SystemMessage(content="You are an expert academic summarizer."),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def compare_papers(self, papers: list[Paper]) -> str:
        """
        Compare multiple papers and identify common themes and differences.

        Args:
            papers: List of papers to compare.

        Returns:
            Comparison analysis.
        """
        if len(papers) < 2:
            return "Need at least 2 papers to compare."

        paper_info = []
        for i, paper in enumerate(papers, 1):
            info = f"""Paper {i}:
Title: {paper.title}
Authors: {', '.join(paper.authors[:3])}
Year: {paper.year or 'Unknown'}
Citations: {paper.citation_count or 'Unknown'}
Abstract: {paper.abstract or 'Not available'}
"""
            paper_info.append(info)

        prompt = f"""Compare the following academic papers:

{chr(10).join(paper_info)}

Provide an analysis that includes:
1. Common themes and research areas
2. Methodological differences
3. Complementary findings
4. Gaps or contradictions
5. Recommendations for further reading
"""

        messages = [
            SystemMessage(content="You are an expert academic analyst."),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def generate_literature_review(
        self,
        topic: str,
        num_papers: int = 10,
    ) -> str:
        """
        Generate a mini literature review on a topic.

        Args:
            topic: Research topic to review.
            num_papers: Number of papers to include.

        Returns:
            Literature review text.
        """
        # Search for papers
        papers = self.search_papers(topic, max_results=num_papers)

        if not papers:
            return f"No papers found for topic: {topic}"

        # Build paper context
        paper_context = []
        for paper in papers:
            ctx = f"""- {paper.title}
  Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}
  Year: {paper.year or 'Unknown'}
  Venue: {paper.venue or 'Unknown'}
  Citations: {paper.citation_count or 'Unknown'}
  Abstract: {paper.abstract or 'Not available'}
"""
            paper_context.append(ctx)

        prompt = f"""Generate a literature review on the topic: "{topic}"

Based on these papers:
{chr(10).join(paper_context)}

Structure the review as follows:
1. Introduction to the research area
2. Key themes and approaches in the literature
3. Notable findings and contributions
4. Research gaps and future directions
5. Conclusion

Include proper citations in the format: (Author et al., Year)
"""

        messages = [
            SystemMessage(content="You are an expert academic writer specializing in literature reviews."),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []

    def clear_documents(self) -> None:
        """Clear all loaded documents."""
        self.loaded_documents = {}
        self.vector_store = None
        self._update_context()

    def get_loaded_documents_info(self) -> list[dict]:
        """Get information about loaded documents."""
        info = []
        for name, doc in self.loaded_documents.items():
            info.append({
                "name": name,
                "title": doc.metadata.title,
                "author": doc.metadata.author,
                "pages": doc.metadata.num_pages,
                "tables": sum(len(p.tables) for p in doc.pages),
            })
        return info
