"""
Document loaders (Step 5, FR-1).

Each loader takes a file path and returns raw text - no chunking here,
that's chunking.py's job. Dispatch is by file extension via
load_document(), which is what pipeline.py (Step 6) actually calls.
"""

from pathlib import Path

from pypdf import PdfReader


def load_txt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_markdown(path: str | Path) -> str:
    
    return Path(path).read_text(encoding="utf-8")


def load_pdf(path: str | Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


_LOADERS = {
    ".txt": load_txt,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".pdf": load_pdf,
}


def load_document(path: str | Path) -> str:
    """
    Dispatches to the right loader based on file extension.
    Raises ValueError for unsupported extensions (FR-1 scopes this to
    PDF, TXT, and Markdown only).
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in _LOADERS:
        raise ValueError(
            f"Unsupported document type '{ext}' for {path}. "
            f"Supported: {sorted(_LOADERS.keys())}"
        )
    return _LOADERS[ext](path)

if __name__ == "__main__":
    from pathlib import Path

    corpus_dir = Path("data/corpus")

    # 1. Check folder exists
    if not corpus_dir.exists():
        print(f"Directory not found: {corpus_dir.resolve()}")
    else:
        # 2. Find all .md files
        md_files = list(corpus_dir.glob("*.md"))
        print(f"Found {len(md_files)} Markdown files in {corpus_dir}\n")

        if md_files:
            # 3. Load the first file as a test
            test_file = md_files[0]
            print(f"--- Loading: {test_file.name} ---")
            raw_content = load_document(test_file)
            
            print(f"File Size: {len(raw_content)} characters")
            print("Preview (First 200 chars):")
            print("-" * 40)
            print(raw_content[:200])
            print("-" * 40)
            print("\nSuccessfully loaded using load_document()!")
        else:
            print("No .md files found in data/corpus/")