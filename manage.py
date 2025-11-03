"""CLI for managing models and vector store."""
import shutil
import json
from pathlib import Path
import click

from config import DATA_DIR, VECTOR_DB_PATH, DOCUMENTS_DIR, SESSIONS_DIR
from model_manager import ModelManager


@click.group()
def cli():
    """Management commands for AskDocs-AI."""
    pass


@cli.command()
def download_models():
    """Download required models for offline use."""
    manager = ModelManager()
    info = manager.ensure_models()
    click.echo(json.dumps(info, indent=2))


@cli.command()
@click.option("--out", "out_dir", type=click.Path(), default=str(DATA_DIR / "backups"))
def backup(out_dir):
    """Backup vector store and sessions."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Copy vector DB and sessions
    for p in [VECTOR_DB_PATH, SESSIONS_DIR]:
        if p.exists():
            dest = out / p.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(p, dest)
    click.echo(f"Backup completed at {out}")


@cli.command()
def rebuild_index():
    """Rebuild vector store from documents."""
    # Lazy import to avoid heavy deps at CLI import time
    from vector_store import VectorStore
    from document_manager import DocumentManager
    from parsers.document_parser import DocumentParser

    store = VectorStore()
    manager = DocumentManager(store)

    for file_path in DOCUMENTS_DIR.glob("**/*"):
        if file_path.is_file():
            parser = DocumentParser.get_parser_for_file(file_path)
            if parser:
                click.echo(f"Indexing {file_path.name}...")
                manager.add_document(file_path)

    click.echo("Rebuild completed.")


if __name__ == "__main__":
    cli()


