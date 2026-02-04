"""Tests for the PDF Loader module."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from academic_assistant.pdf_loader import PDFLoader, PDFDocument, PDFMetadata, PDFPage


class TestPDFMetadata:
    """Tests for PDFMetadata model."""

    def test_default_values(self):
        """Test default values for metadata."""
        metadata = PDFMetadata()
        assert metadata.title is None
        assert metadata.author is None
        assert metadata.num_pages == 0

    def test_with_values(self):
        """Test metadata with values."""
        metadata = PDFMetadata(
            title="Test Paper",
            author="John Doe",
            num_pages=10,
        )
        assert metadata.title == "Test Paper"
        assert metadata.author == "John Doe"
        assert metadata.num_pages == 10


class TestPDFPage:
    """Tests for PDFPage model."""

    def test_default_values(self):
        """Test default values for page."""
        page = PDFPage(page_number=1, text="Sample text")
        assert page.page_number == 1
        assert page.text == "Sample text"
        assert page.tables == []
        assert page.metadata == {}

    def test_with_tables(self):
        """Test page with tables."""
        tables = [[["Header 1", "Header 2"], ["Value 1", "Value 2"]]]
        page = PDFPage(page_number=1, text="Sample text", tables=tables)
        assert len(page.tables) == 1
        assert page.tables[0][0] == ["Header 1", "Header 2"]


class TestPDFDocument:
    """Tests for PDFDocument model."""

    def test_get_text(self):
        """Test getting full text from document."""
        pages = [
            PDFPage(page_number=1, text="Page 1 content"),
            PDFPage(page_number=2, text="Page 2 content"),
        ]
        doc = PDFDocument(
            metadata=PDFMetadata(num_pages=2),
            pages=pages,
        )
        text = doc.get_text()
        assert "Page 1 content" in text
        assert "Page 2 content" in text

    def test_get_text_with_full_text(self):
        """Test getting full text when already set."""
        doc = PDFDocument(
            metadata=PDFMetadata(num_pages=1),
            pages=[],
            full_text="Pre-extracted text",
        )
        assert doc.get_text() == "Pre-extracted text"

    def test_get_page_text(self):
        """Test getting text from specific page."""
        pages = [
            PDFPage(page_number=1, text="Page 1 content"),
            PDFPage(page_number=2, text="Page 2 content"),
        ]
        doc = PDFDocument(
            metadata=PDFMetadata(num_pages=2),
            pages=pages,
        )
        assert doc.get_page_text(1) == "Page 1 content"
        assert doc.get_page_text(2) == "Page 2 content"
        assert doc.get_page_text(3) is None


class TestPDFLoader:
    """Tests for PDFLoader class."""

    def test_initialization_defaults(self):
        """Test loader initialization with defaults."""
        loader = PDFLoader()
        assert loader.chunk_size > 0
        assert loader.chunk_overlap >= 0

    def test_initialization_custom(self):
        """Test loader initialization with custom values."""
        loader = PDFLoader(chunk_size=500, chunk_overlap=50, max_pages=5)
        assert loader.chunk_size == 500
        assert loader.chunk_overlap == 50
        assert loader.max_pages == 5

    def test_load_from_path_file_not_found(self):
        """Test loading non-existent file."""
        loader = PDFLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_from_path("/nonexistent/file.pdf")

    def test_load_from_path_not_pdf(self, tmp_path):
        """Test loading non-PDF file."""
        text_file = tmp_path / "test.txt"
        text_file.write_text("Not a PDF")

        loader = PDFLoader()
        with pytest.raises(ValueError):
            loader.load_from_path(text_file)

    @patch("academic_assistant.pdf_loader.pdfplumber")
    @patch("academic_assistant.pdf_loader.PdfReader")
    def test_load_from_stream(self, mock_reader_class, mock_pdfplumber):
        """Test loading from stream."""
        # Setup mocks
        mock_reader = MagicMock()
        mock_reader.metadata = {"/Title": "Test", "/Author": "Author"}
        mock_reader.pages = [MagicMock(), MagicMock()]
        mock_reader_class.return_value = mock_reader

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page content"
        mock_page.extract_tables.return_value = []

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page, mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        loader = PDFLoader()
        stream = io.BytesIO(b"fake pdf content")
        doc = loader.load_from_stream(stream, "test.pdf")

        assert doc.metadata.title == "Test"
        assert doc.metadata.author == "Author"
        assert len(doc.pages) == 2

    def test_get_document_summary_file_not_found(self):
        """Test summary for non-existent file."""
        loader = PDFLoader()
        with pytest.raises(FileNotFoundError):
            loader.get_document_summary("/nonexistent/file.pdf")
