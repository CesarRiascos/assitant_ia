"""Command Line Interface for Academic Assistant.

This module provides a CLI for interacting with the Academic Assistant.
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from academic_assistant.assistant import AcademicAssistant
from academic_assistant.pdf_loader import PDFLoader
from academic_assistant.research_apis import MultiSourceSearcher

console = Console()


def print_welcome():
    """Print welcome message."""
    console.print(
        Panel.fit(
            "[bold blue]Academic Research Assistant[/bold blue]\n"
            "AI-powered assistant for academic investigations\n\n"
            "Commands:\n"
            "  [green]search <query>[/green] - Search academic papers\n"
            "  [green]load <pdf_path>[/green] - Load a PDF document\n"
            "  [green]chat <message>[/green] - Chat with the assistant\n"
            "  [green]summary[/green] - Summarize loaded documents\n"
            "  [green]review <topic>[/green] - Generate literature review\n"
            "  [green]docs[/green] - List loaded documents\n"
            "  [green]clear[/green] - Clear conversation history\n"
            "  [green]quit[/green] - Exit the assistant\n",
            title="Welcome",
        )
    )


def print_papers(papers):
    """Print papers in a table format."""
    if not papers:
        console.print("[yellow]No papers found.[/yellow]")
        return

    table = Table(title=f"Found {len(papers)} papers")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Title", style="white", max_width=50)
    table.add_column("Authors", style="green", max_width=30)
    table.add_column("Year", style="yellow", width=6)
    table.add_column("Citations", style="magenta", width=10)
    table.add_column("Source", style="blue", width=12)

    for i, paper in enumerate(papers, 1):
        authors = ", ".join(paper.authors[:2])
        if len(paper.authors) > 2:
            authors += "..."
        table.add_row(
            str(i),
            paper.title[:50] + "..." if len(paper.title) > 50 else paper.title,
            authors[:30] + "..." if len(authors) > 30 else authors,
            str(paper.year) if paper.year else "-",
            str(paper.citation_count) if paper.citation_count else "-",
            paper.source,
        )

    console.print(table)


def print_documents(docs_info):
    """Print loaded documents."""
    if not docs_info:
        console.print("[yellow]No documents loaded.[/yellow]")
        return

    table = Table(title="Loaded Documents")
    table.add_column("Name", style="cyan")
    table.add_column("Title", style="white", max_width=40)
    table.add_column("Author", style="green", max_width=25)
    table.add_column("Pages", style="yellow")
    table.add_column("Tables", style="magenta")

    for doc in docs_info:
        table.add_row(
            doc["name"],
            doc["title"] or "-",
            doc["author"] or "-",
            str(doc["pages"]),
            str(doc["tables"]),
        )

    console.print(table)


def interactive_mode():
    """Run the assistant in interactive mode."""
    print_welcome()

    try:
        assistant = AcademicAssistant()
    except Exception as e:
        console.print(f"[red]Error initializing assistant: {e}[/red]")
        console.print("[yellow]Make sure OPENAI_API_KEY is set in your environment.[/yellow]")
        return

    while True:
        try:
            user_input = console.input("\n[bold cyan]You>[/bold cyan] ").strip()

            if not user_input:
                continue

            # Parse commands
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if command in ["quit", "exit", "q"]:
                console.print("[blue]Goodbye![/blue]")
                break

            elif command == "search":
                if not args:
                    console.print("[red]Please provide a search query.[/red]")
                    continue
                with console.status("Searching academic databases..."):
                    papers = assistant.search_papers(args, max_results=10)
                print_papers(papers)

            elif command == "load":
                if not args:
                    console.print("[red]Please provide a PDF file path.[/red]")
                    continue
                path = Path(args)
                if not path.exists():
                    console.print(f"[red]File not found: {path}[/red]")
                    continue
                with console.status("Loading PDF..."):
                    doc = assistant.load_pdf(path)
                console.print(f"[green]Loaded: {doc.metadata.title or path.name}[/green]")
                console.print(f"  Pages: {doc.metadata.num_pages}")
                console.print(f"  Author: {doc.metadata.author or 'Unknown'}")

            elif command == "chat":
                if not args:
                    console.print("[red]Please provide a message.[/red]")
                    continue
                with console.status("Thinking..."):
                    response = assistant.chat(args)
                console.print("\n[bold green]Assistant>[/bold green]")
                console.print(Markdown(response.message))
                if response.suggestions:
                    console.print("\n[dim]Suggestions:[/dim]")
                    for sug in response.suggestions:
                        console.print(f"  [dim]- {sug}[/dim]")

            elif command == "summary":
                docs = assistant.get_loaded_documents_info()
                if not docs:
                    console.print("[yellow]No documents loaded. Use 'load <path>' first.[/yellow]")
                    continue
                for doc in docs:
                    with console.status(f"Summarizing {doc['name']}..."):
                        summary = assistant.summarize_document(doc["name"])
                    console.print(f"\n[bold]Summary of {doc['name']}:[/bold]")
                    console.print(Markdown(summary))

            elif command == "review":
                if not args:
                    console.print("[red]Please provide a topic.[/red]")
                    continue
                with console.status("Generating literature review..."):
                    review = assistant.generate_literature_review(args)
                console.print("\n[bold]Literature Review:[/bold]")
                console.print(Markdown(review))

            elif command == "docs":
                docs = assistant.get_loaded_documents_info()
                print_documents(docs)

            elif command == "clear":
                assistant.clear_history()
                console.print("[green]Conversation history cleared.[/green]")

            elif command == "help":
                print_welcome()

            else:
                # Treat as a chat message
                with console.status("Thinking..."):
                    response = assistant.chat(user_input)
                console.print("\n[bold green]Assistant>[/bold green]")
                console.print(Markdown(response.message))
                if response.suggestions:
                    console.print("\n[dim]Suggestions:[/dim]")
                    for sug in response.suggestions:
                        console.print(f"  [dim]- {sug}[/dim]")

        except KeyboardInterrupt:
            console.print("\n[blue]Goodbye![/blue]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def search_command(args):
    """Handle search command."""
    searcher = MultiSourceSearcher()

    sources = None
    if args.source:
        sources = [args.source]

    with console.status("Searching..."):
        if sources:
            results = searcher.search(args.query, sources, args.limit)
            papers = []
            for result in results.values():
                papers.extend(result.papers)
        else:
            papers = searcher.search_all_and_deduplicate(args.query, args.limit)

    print_papers(papers)


def load_command(args):
    """Handle load command."""
    loader = PDFLoader()

    try:
        doc = loader.load_from_path(args.file)
        console.print(f"[green]Successfully loaded: {args.file}[/green]")
        console.print(f"  Title: {doc.metadata.title or 'Unknown'}")
        console.print(f"  Author: {doc.metadata.author or 'Unknown'}")
        console.print(f"  Pages: {doc.metadata.num_pages}")

        if args.summary:
            summary = loader.get_document_summary(args.file)
            console.print(f"  Characters: {summary['total_characters']}")
            console.print(f"  Tables: {summary['total_tables']}")

        if args.output:
            with open(args.output, "w") as f:
                f.write(doc.get_text())
            console.print(f"[green]Text extracted to: {args.output}[/green]")

    except FileNotFoundError:
        console.print(f"[red]File not found: {args.file}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Academic Research Assistant - AI-powered academic investigation tool"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Interactive mode
    interactive_parser = subparsers.add_parser(
        "interactive", aliases=["i"], help="Start interactive mode"
    )

    # Search command
    search_parser = subparsers.add_parser("search", aliases=["s"], help="Search academic papers")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "-n", "--limit", type=int, default=10, help="Maximum number of results"
    )
    search_parser.add_argument(
        "--source",
        choices=["semantic_scholar", "arxiv", "crossref", "openalex"],
        help="Specific source to search",
    )

    # Load command
    load_parser = subparsers.add_parser("load", aliases=["l"], help="Load and process a PDF")
    load_parser.add_argument("file", help="Path to PDF file")
    load_parser.add_argument("-o", "--output", help="Output file for extracted text")
    load_parser.add_argument(
        "-s", "--summary", action="store_true", help="Show document summary"
    )

    args = parser.parse_args()

    if args.command in ["interactive", "i"] or args.command is None:
        interactive_mode()
    elif args.command in ["search", "s"]:
        search_command(args)
    elif args.command in ["load", "l"]:
        load_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
