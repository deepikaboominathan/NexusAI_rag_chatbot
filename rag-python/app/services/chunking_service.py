
import re


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[str]:
    """
    Split text into overlapping chunks while preferring
    paragraph and sentence boundaries.

    chunk_size and chunk_overlap are measured in characters.
    """

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and "
            "smaller than chunk_size"
        )

    # Normalize whitespace without removing paragraph boundaries.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        return []

    # Prefer paragraphs, then sentences, then smaller pieces.
    units = re.split(
        r"(\n\n+|(?<=[.!?])\s+)",
        text,
    )

    segments = []
    current = ""

    for unit in units:
        if not unit:
            continue

        candidate = current + unit

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current.strip():
            segments.append(current.strip())

        # Handle an individual paragraph or sentence that is
        # longer than the maximum chunk size.
        if len(unit) > chunk_size:
            start = 0

            while start < len(unit):
                piece = unit[start:start + chunk_size].strip()

                if piece:
                    segments.append(piece)

                start += chunk_size - chunk_overlap

            current = ""
        else:
            current = unit

    if current.strip():
        segments.append(current.strip())

    # Merge small adjacent segments where possible.
    chunks = []
    current = ""

    for segment in segments:
        if not current:
            current = segment
        elif len(current) + 2 + len(segment) <= chunk_size:
            current += "\n\n" + segment
        else:
            chunks.append(current.strip())

            # Preserve context from the previous chunk.
            overlap = ""

            if chunk_overlap:
                overlap_text = current[-chunk_overlap:]
                first_space = overlap_text.find(" ")

                if first_space != -1:
                    overlap = overlap_text[first_space + 1:].strip()
            current = (
                overlap + "\n\n" + segment
                if overlap
                else segment
            )

            # Keep the configured maximum chunk size.
            if len(current) > chunk_size:
                current = current[-chunk_size:]

    if current.strip():
        chunks.append(current.strip())

    return chunks
