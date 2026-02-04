"""
Academic Assistant - AI Generative Assistant for Academic Investigations.

This package provides tools for:
- Loading and processing PDF documents
- Searching academic research APIs (Semantic Scholar, arXiv, CrossRef, OpenAlex)
- AI-powered research assistance using LangChain
"""

__version__ = "0.1.0"
__author__ = "Academic Assistant Team"

from academic_assistant.pdf_loader import PDFLoader
from academic_assistant.research_apis import (
    ArxivClient,
    CrossRefClient,
    OpenAlexClient,
    ResearchAPIClient,
    SemanticScholarClient,
)
from academic_assistant.assistant import AcademicAssistant
from academic_assistant.config import Settings

__all__ = [
    "PDFLoader",
    "ArxivClient",
    "CrossRefClient",
    "OpenAlexClient",
    "ResearchAPIClient",
    "SemanticScholarClient",
    "AcademicAssistant",
    "Settings",
]
