"""Test 2 of §13 — upload → search smoke.

POST /text with distinctive content, then GET /search with a query that
should retrieve it. Asserts the upload flows through chunking, embedding,
and persistence correctly, and that hybrid retrieval + rerank surfaces the
uploaded document at the top.
"""

from uuid import uuid4

from fastapi.testclient import TestClient

from .conftest import requires_voyage


@requires_voyage
def test_upload_then_search_returns_uploaded_document(
    client: TestClient,
    db_cleanup: None,
) -> None:
    # Distinctive proper noun + unusual phrasing so neither the dense nor
    # lexical lane has to fight ambient corpus noise. uuid salt → unique
    # SHA-256 per run, so /text always exercises chunk→embed→persist
    # instead of short-circuiting through §12 dedupe on leftover rows.
    payload = {
        "title": "Zorbnak quarterly notes",
        "text": (
            f"[run {uuid4()}] "
            "The Zorbnak fiscal review for Q3 2025 reports a 47% increase in "
            "thaumaturgical widget sales across the Belgrade office. The "
            "Zorbnak board attributes this to improved supply-chain "
            "coordination with the Antwerp distribution hub."
        ),
        "author": "Smoke Test",
        "metadata": {"type": "test"},
    }

    upload = client.post("/text", json=payload)
    assert upload.status_code == 200, upload.text
    document_id = upload.json()["document_id"]

    response = client.get("/search", params={"q": "Zorbnak Belgrade widget sales"})
    assert response.status_code == 200, response.text

    results = response.json()["results"]
    assert results, "search returned no results for the just-uploaded doc"
    assert results[0]["document_id"] == document_id, (
        f"top result was {results[0]['document_id']}, expected {document_id}"
    )
