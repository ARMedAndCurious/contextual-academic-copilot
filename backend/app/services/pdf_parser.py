import pdfplumber
import re


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract raw text from a PDF using pdfplumber.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Raw extracted text.
    """

    all_text = []

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()

                if text:
                    all_text.append(text)

    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

    return "\n".join(all_text)


def clean_text(text: str) -> str:
    """
    Cleans PDF text to make it suitable for chunking and embeddings.
    """

    # Remove excessive newlines
    text = re.sub(r'\n{2,}', '\n', text)

    # Fix hyphenated line breaks (very common in research papers)
    # Example:
    # "environ-\nmental" -> "environmental"
    text = re.sub(r'-\n(\w+)', r'\1', text)

    # Replace remaining newlines with spaces
    # (important: embeddings work on sentences, not broken lines)
    text = re.sub(r'\n', ' ', text)

    # Remove multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)

    # Remove page numbers like "Page 1", "1", etc.
    text = re.sub(r'\bPage \d+\b', '', text)
    text = re.sub(r'\b\d+\b(?=\s*$)', '', text)

    return text.strip()


def parse_pdf(file_path: str) -> str:
    """
    Main function your system will use.
    """

    raw_text = extract_text_from_pdf(file_path)
    cleaned_text = clean_text(raw_text)

    if len(cleaned_text) < 500:
        raise Exception("PDF extraction failed — too little text extracted.")

    return cleaned_text