"""PDF Loading and Processing Module.

This module provides functionality to load, parse, and process PDF documents
for use in academic research assistance.
"""

import io
from pathlib import Path
from typing import BinaryIO, Iterator, Optional, Union

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from pypdf import PdfReader

from academic_assistant.config import settings


class PDFMetadata(BaseModel):
    """Metadata extracted from a PDF document."""

    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    num_pages: int = 0
    file_path: Optional[str] = None


class PDFPage(BaseModel):
    """Represents a single page from a PDF document."""

    page_number: int
    text: str
    tables: list[list[list[str]]] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class PDFDocument(BaseModel):
    """Represents a complete PDF document."""

    metadata: PDFMetadata
    pages: list[PDFPage] = Field(default_factory=list)
    full_text: str = ""

    def get_text(self) -> str:
        """Get the full text of the document."""
        if self.full_text:
            return self.full_text
        return "\n\n".join(page.text for page in self.pages)

    def get_page_text(self, page_number: int) -> Optional[str]:
        """Get text from a specific page (1-indexed)."""
        for page in self.pages:
            if page.page_number == page_number:
                return page.text
        return None


class PDFLoader:
    """
    PDF Loader for academic documents.

    Supports loading PDFs from file paths or binary streams,
    extracting text, tables, and metadata.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        max_pages: Optional[int] = None,
    ):
        """
        Initialize the PDF Loader.

        Args:
            chunk_size: Size of text chunks for splitting. Defaults to config value.
            chunk_overlap: Overlap between chunks. Defaults to config value.
            max_pages: Maximum number of pages to process. None means all pages.
        """
        self.chunk_size = chunk_size or settings.pdf.chunk_size
        self.chunk_overlap = chunk_overlap or settings.pdf.chunk_overlap
        self.max_pages = max_pages or settings.pdf.max_pages

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _extract_metadata(self, reader: PdfReader, file_path: Optional[str] = None) -> PDFMetadata:
        """Extract metadata from a PDF reader object."""
        info = reader.metadata or {}

        return PDFMetadata(
            title=info.get("/Title"),
            author=info.get("/Author"),
            subject=info.get("/Subject"),
            creator=info.get("/Creator"),
            producer=info.get("/Producer"),
            creation_date=str(info.get("/CreationDate")) if info.get("/CreationDate") else None,
            modification_date=str(info.get("/ModDate")) if info.get("/ModDate") else None,
            num_pages=len(reader.pages),
            file_path=file_path,
        )

    def load_from_path(self, file_path: Union[str, Path]) -> PDFDocument:
        """
        Load a PDF document from a file path.

        Args:
            file_path: Path to the PDF file.

        Returns:
            PDFDocument with extracted content and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a valid PDF.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not file_path.suffix.lower() == ".pdf":
            raise ValueError(f"File is not a PDF: {file_path}")

        with open(file_path, "rb") as f:
            return self.load_from_stream(f, str(file_path))

    def load_from_stream(
        self, stream: BinaryIO, source_name: Optional[str] = None
    ) -> PDFDocument:
        """
        Load a PDF document from a binary stream.

        Args:
            stream: Binary stream containing PDF data.
            source_name: Optional name for the source (for metadata).

        Returns:
            PDFDocument with extracted content and metadata.
        """
        # Read stream content
        content = stream.read()
        stream.seek(0)

        # Extract metadata using pypdf
        reader = PdfReader(io.BytesIO(content))
        metadata = self._extract_metadata(reader, source_name)

        # Extract text and tables using pdfplumber
        pages: list[PDFPage] = []
        full_text_parts: list[str] = []

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            max_pages = self.max_pages or len(pdf.pages)
            for i, page in enumerate(pdf.pages[:max_pages]):
                page_number = i + 1
                text = page.extract_text() or ""
                full_text_parts.append(text)

                # Extract tables
                tables = []
                for table in page.extract_tables():
                    if table:
                        # Convert None values to empty strings
                        clean_table = [
                            [cell if cell is not None else "" for cell in row]
                            for row in table
                        ]
                        tables.append(clean_table)

                pages.append(
                    PDFPage(
                        page_number=page_number,
                        text=text,
                        tables=tables,
                        metadata={"source": source_name, "page": page_number},
                    )
                )

        return PDFDocument(
            metadata=metadata,
            pages=pages,
            full_text="\n\n".join(full_text_parts),
        )

    def load_from_bytes(self, data: bytes, source_name: Optional[str] = None) -> PDFDocument:
        """
        Load a PDF document from bytes.

        Args:
            data: PDF content as bytes.
            source_name: Optional name for the source.

        Returns:
            PDFDocument with extracted content and metadata.
        """
        return self.load_from_stream(io.BytesIO(data), source_name)

    def load_and_split(
        self, file_path: Union[str, Path]
    ) -> list[Document]:
        """
        Load a PDF and split it into LangChain Documents.

        This is useful for vector store ingestion.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of LangChain Document objects.
        """
        pdf_doc = self.load_from_path(file_path)
        documents = []

        for page in pdf_doc.pages:
            doc = Document(
                page_content=page.text,
                metadata={
                    "source": str(file_path),
                    "page": page.page_number,
                    "title": pdf_doc.metadata.title,
                    "author": pdf_doc.metadata.author,
                },
            )
            documents.append(doc)

        # Split documents into chunks
        return self.text_splitter.split_documents(documents)

    def iter_pages(
        self, file_path: Union[str, Path]
    ) -> Iterator[PDFPage]:
        """
        Iterate over pages in a PDF document.

        Args:
            file_path: Path to the PDF file.

        Yields:
            PDFPage objects for each page.
        """
        pdf_doc = self.load_from_path(file_path)
        yield from pdf_doc.pages

    def extract_tables(self, file_path: Union[str, Path]) -> list[list[list[str]]]:
        """
        Extract all tables from a PDF document.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of tables, where each table is a list of rows.
        """
        pdf_doc = self.load_from_path(file_path)
        tables = []
        for page in pdf_doc.pages:
            tables.extend(page.tables)
        return tables

    def get_document_summary(self, file_path: Union[str, Path]) -> dict:
        """
        Get a summary of a PDF document.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Dictionary with document summary information.
        """
        pdf_doc = self.load_from_path(file_path)
        total_tables = sum(len(page.tables) for page in pdf_doc.pages)
        total_chars = sum(len(page.text) for page in pdf_doc.pages)

        return {
            "title": pdf_doc.metadata.title,
            "author": pdf_doc.metadata.author,
            "num_pages": pdf_doc.metadata.num_pages,
            "total_characters": total_chars,
            "total_tables": total_tables,
            "file_path": str(file_path),
        }
