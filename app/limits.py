"""Project-wide ingestion limits — the load-bearing constants from DECISIONS.md §17.

Lives outside `app.settings` because these are tied to embedding economics and
worker memory, not per-deployment knobs. If a real customer uploads content
that hits either cap, edit this file; do not surface them through env vars.
"""

from typing import Annotated

from pydantic import Field

# Multipart body cap on POST /document (DECISIONS.md §17). Enforced by the
# global body-size middleware before the multipart body is parsed.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Extracted-text cap on both POST /text and POST /document (DECISIONS.md §17).
# At ~4 chars/token this is ~750K tokens (~1,400 chunks at 600 tokens each).
MAX_TEXT_CHARS = 3_000_000

# Reusable Pydantic-validated string for any ingestion text input. Used by
# IngestTextRequest.text, and by /document's post-extraction re-validation
# via TypeAdapter(MaxText).validate_python(text).
MaxText = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_CHARS)]
