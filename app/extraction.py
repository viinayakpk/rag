"""PDF text extraction (DECISIONS.md §5).

One function: `extract_text(pdf_bytes) -> str`. Walks every page, joins them
with paragraph breaks (`\\n\\n`) so the chunker's top-level separator stays
meaningful. Pure: takes bytes, returns text. The route owns HTTP-status
mapping and the magic-byte pre-check.

Migration story: swapping pymupdf → pypdf is a same-day replacement of this
file's body — the function signature stays.
"""

import pymupdf

from app.limits import MAX_TEXT_CHARS


class TextTooLargeError(Exception):
    """Extracted text exceeded MAX_TEXT_CHARS during page-by-page extraction."""


def extract_text(pdf_bytes: bytes) -> str:
    """Extract concatenated plain text from every page of a PDF.

    Caps cumulative extracted characters at `MAX_TEXT_CHARS` page-by-page so a
    PDF whose compressed text streams expand far beyond the file's byte size
    can't drive the worker into RAM blow-up before the post-extraction text
    cap fires. Raises `TextTooLargeError` the moment the running total crosses
    the cap; the route maps it to HTTP 422.

    Raises `pymupdf.FileDataError` if the PDF is corrupt or malformed; the
    route catches this and maps to HTTP 400. The caller is responsible for
    the `%PDF-` magic-byte check before invoking this function.
    """
    parts: list[str] = []
    cumulative = 0
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            page_text = page.get_text()
            # The final string is "\n\n".join(parts) — account for the 2-char
            # join inserted between every adjacent pair of parts.
            cumulative += len(page_text) + (2 if parts else 0)
            if cumulative > MAX_TEXT_CHARS:
                raise TextTooLargeError(
                    f"extracted text exceeds {MAX_TEXT_CHARS}-character cap"
                )
            parts.append(page_text)
    return "\n\n".join(parts)
