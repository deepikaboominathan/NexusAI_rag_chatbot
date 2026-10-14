
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def extract_text(file_path: str | Path) -> str:
    """
    Extract text from a PDF, TXT, or DOCX file.

    Args:
        file_path: Path to the saved document.

    Returns:
        Extracted text with surrounding whitespace removed.

    Raises:
        ValueError: If the file type is unsupported or no text is found.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension or 'unknown'}. "
            "Supported types are PDF, TXT, and DOCX."
        )

    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path.name}")

    if extension == ".txt":
        text = path.read_text(encoding="utf-8-sig", errors="replace")

    elif extension == ".pdf":
        reader = PdfReader(str(path))
        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        text = "\n".join(pages)

    else:  # .docx
        document = DocxDocument(str(path))
        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]
        text = "\n".join(paragraphs)

    text = text.strip()

    if not text:
        raise ValueError(
            "No extractable text was found. "
            "The document may be empty, scanned, or image-only."
        )

    return text
