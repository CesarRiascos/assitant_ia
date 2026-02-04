# Academic Research Assistant

An AI-powered generative assistant for academic investigations. This tool helps researchers with literature review, document analysis, and research synthesis by integrating multiple academic APIs and LLM capabilities.

## Features

### PDF Document Processing
- Load and parse PDF documents
- Extract text, metadata, and tables
- Chunk documents for vector search
- Support for multiple document analysis

### Academic Research APIs
Search across multiple academic databases with public APIs:
- **Semantic Scholar** - Comprehensive academic paper database with citation data
- **arXiv** - Preprint server for physics, mathematics, computer science, and more
- **CrossRef** - DOI registration agency with metadata for scholarly content
- **OpenAlex** - Open catalog of scholarly works, authors, venues, and institutions

### AI-Powered Assistance
- Literature review generation
- Document summarization
- Research synthesis across multiple papers
- Citation generation and formatting
- Research question formulation

## Installation

### Prerequisites
- Python 3.9 or higher
- OpenAI API key (for AI features)

### Install from source

```bash
# Clone the repository
git clone https://github.com/CesarRiascos/assitant_ia.git
cd assitant_ia

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Configuration

Create a `.env` file in the project root (or copy from `.env.example`):

```bash
cp .env.example .env
```

Edit the `.env` file with your API keys:

```env
# Required for AI features
OPENAI_API_KEY=your_openai_api_key_here

# Optional - for higher rate limits on Semantic Scholar
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key

# Optional - for polite pool (faster responses) on CrossRef/OpenAlex
CROSSREF_EMAIL=your_email@example.com
```

## Usage

### Command Line Interface

#### Interactive Mode

Start an interactive session with the assistant:

```bash
academic-assistant
# or
academic-assistant interactive
```

Available commands in interactive mode:
- `search <query>` - Search academic papers
- `load <pdf_path>` - Load a PDF document
- `chat <message>` - Chat with the assistant
- `summary` - Summarize loaded documents
- `review <topic>` - Generate a literature review
- `docs` - List loaded documents
- `clear` - Clear conversation history
- `quit` - Exit the assistant

#### Search Command

Search academic papers from the command line:

```bash
# Search all sources
academic-assistant search "machine learning transformers"

# Search specific source
academic-assistant search "deep learning" --source arxiv

# Limit results
academic-assistant search "neural networks" -n 5
```

#### Load Command

Process PDF documents:

```bash
# Load and display info
academic-assistant load paper.pdf

# Extract text to file
academic-assistant load paper.pdf -o extracted.txt

# Show summary
academic-assistant load paper.pdf --summary
```

### Python API

#### Basic Usage

```python
from academic_assistant import AcademicAssistant

# Initialize the assistant
assistant = AcademicAssistant()

# Search for papers
papers = assistant.search_papers("machine learning", max_results=10)
for paper in papers:
    print(f"{paper.title} ({paper.year}) - {paper.citation_count} citations")

# Chat with the assistant
response = assistant.chat("What are the latest advances in transformer architectures?")
print(response.message)
```

#### PDF Processing

```python
from academic_assistant import PDFLoader

# Initialize loader
loader = PDFLoader()

# Load a PDF
doc = loader.load_from_path("research_paper.pdf")

# Access metadata
print(f"Title: {doc.metadata.title}")
print(f"Author: {doc.metadata.author}")
print(f"Pages: {doc.metadata.num_pages}")

# Get full text
text = doc.get_text()

# Extract tables
tables = loader.extract_tables("research_paper.pdf")

# Split for vector store
chunks = loader.load_and_split("research_paper.pdf")
```

#### Research APIs

```python
from academic_assistant import (
    SemanticScholarClient,
    ArxivClient,
    CrossRefClient,
    OpenAlexClient,
)

# Semantic Scholar
ss_client = SemanticScholarClient()
results = ss_client.search("attention mechanism", max_results=5)

# arXiv
arxiv_client = ArxivClient()
results = arxiv_client.search("quantum computing", max_results=5)

# CrossRef
crossref_client = CrossRefClient()
results = crossref_client.search("climate change", max_results=5)

# OpenAlex
openalex_client = OpenAlexClient()
results = openalex_client.search("CRISPR", max_results=5)
```

#### Multi-Source Search

```python
from academic_assistant.research_apis import MultiSourceSearcher

searcher = MultiSourceSearcher()

# Search specific sources
results = searcher.search(
    "artificial intelligence",
    sources=["semantic_scholar", "arxiv"],
    max_results_per_source=5
)

# Search all sources and deduplicate
papers = searcher.search_all_and_deduplicate(
    "neural networks",
    max_results=20
)
```

#### Document Analysis

```python
from academic_assistant import AcademicAssistant

assistant = AcademicAssistant()

# Load multiple PDFs
assistant.load_pdf("paper1.pdf", name="paper1")
assistant.load_pdf("paper2.pdf", name="paper2")

# Summarize a document
summary = assistant.summarize_document("paper1")

# Compare papers
papers = assistant.search_papers("attention mechanisms", max_results=3)
comparison = assistant.compare_papers(papers)

# Generate literature review
review = assistant.generate_literature_review("transformer architectures", num_papers=10)
```

#### Async Support

```python
import asyncio
from academic_assistant import AcademicAssistant

async def main():
    assistant = AcademicAssistant()
    
    # Async paper search
    papers = await assistant.async_search_papers(
        "deep learning",
        sources=["semantic_scholar", "arxiv"],
        max_results=10
    )
    
    for paper in papers:
        print(paper.title)

asyncio.run(main())
```

## Configuration Options

### Model Configuration

```python
from academic_assistant import AcademicAssistant, Settings

# Custom settings
assistant = AcademicAssistant(
    model_name="gpt-4o",  # or "gpt-4o-mini", "gpt-4-turbo"
    temperature=0.7,
    api_key="your_api_key"  # or use OPENAI_API_KEY env var
)

# Access settings
from academic_assistant.config import settings
print(settings.model.model_name)
print(settings.search.max_results)
```

### PDF Processing Configuration

```python
from academic_assistant import PDFLoader

loader = PDFLoader(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200,    # Overlap between chunks
    max_pages=50          # Maximum pages to process
)
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=academic_assistant

# Run specific test file
pytest tests/test_pdf_loader.py
```

### Code Formatting

```bash
# Format code
black src tests

# Lint code
ruff check src tests

# Type checking
mypy src
```

## Project Structure

```
assitant_ia/
├── src/
│   └── academic_assistant/
│       ├── __init__.py          # Package exports
│       ├── assistant.py         # Main AI assistant
│       ├── cli.py               # Command line interface
│       ├── config.py            # Configuration management
│       ├── pdf_loader.py        # PDF processing
│       └── research_apis.py     # Academic API clients
├── tests/
│   ├── __init__.py
│   ├── test_assistant.py
│   ├── test_pdf_loader.py
│   └── test_research_apis.py
├── .env.example                 # Example environment variables
├── pyproject.toml               # Project configuration
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## API Rate Limits

### Public APIs (No API Key Required)
- **Semantic Scholar**: 100 requests per 5 minutes
- **arXiv**: No strict limit, but be respectful
- **CrossRef**: Polite pool available with email
- **OpenAlex**: Polite pool available with email

### With API Keys
- **Semantic Scholar**: Higher limits with API key
- **OpenAI**: Depends on your plan

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework
- [Semantic Scholar](https://www.semanticscholar.org/) - Academic paper database
- [arXiv](https://arxiv.org/) - Open access preprint server
- [CrossRef](https://www.crossref.org/) - DOI registration and metadata
- [OpenAlex](https://openalex.org/) - Open catalog of scholarly works
