import re
from typing import List


def split_into_sentences(text: str) -> List[str]:
    """
    Splits the document into sentences using punctuation.
    We split on . ! ? followed by a space.
    """

    # Break text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)

    # Remove empty pieces and strip whitespace
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Create sentence-aware overlapping chunks.

    Why we do this:
    - LLMs cannot read full documents
    - We retrieve only relevant chunks later
    - Overlap prevents losing context at boundaries

    Args:
        text: cleaned document text
        chunk_size: approx characters per chunk
        overlap: characters repeated between chunks

    Returns:
        List of text chunks
    """

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    sentences = split_into_sentences(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        # If sentence fits in current chunk → keep adding
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence

        else:
            # Save the current chunk
            chunks.append(current_chunk.strip())

            # Start new chunk WITH overlap
            # take last `overlap` characters from previous chunk
            current_chunk = current_chunk[-overlap:] + " " + sentence

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks