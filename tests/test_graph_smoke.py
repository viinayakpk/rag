"""Test 4 — knowledge-graph extraction + entity-filtered search smoke.

Verifies the full ingest → extract → resolve → persist → entity-filtered
retrieve loop:
  1. Two documents that share an organization but mention different cities
     are ingested. Extraction + resolution run synchronously per ingest.
  2. ?entity=<org> on /search surfaces chunks from both docs (single filter).
  3. ?entity=<org>&entity=<unique-city> AND-filters to chunks from only the
     document that mentions both. The other doc must be absent.
  4. ?entity=<unknown-name> returns empty (under chunk-level AND, an
     unresolvable name yields no chunks — DECISIONS.md §18.7 surface).

Note: entities are corpus-global (not document-scoped, see DECISIONS.md
§18.11 limitations) so cross-test pollution is possible; assertions check
membership of specific UUIDs (not position or set equality), which is
robust to whatever else is in the entities table.
"""

from uuid import uuid4

from fastapi.testclient import TestClient

from .conftest import requires_anthropic, requires_voyage


@requires_voyage
@requires_anthropic
def test_entity_extraction_and_filtered_search(
    client: TestClient,
    db_cleanup: None,
) -> None:
    salt = uuid4().hex[:8]
    org = f"Zorbnak-{salt}"
    city_a = f"Belgrade-{salt}"
    city_b = f"Antwerp-{salt}"

    # Doc A — mentions the org and city_a, not city_b
    upload_a = client.post(
        "/text",
        json={
            "title": f"{org} {city_a} expansion notes",
            "text": (
                f"{org} opened its new {city_a} headquarters in Q1 2026. "
                f"The {city_a} office will lead European operations and employ 200 engineers."
            ),
        },
    )
    assert upload_a.status_code == 200, upload_a.text
    doc_a_id = upload_a.json()["document_id"]

    # Doc B — mentions the org and city_b, not city_a
    upload_b = client.post(
        "/text",
        json={
            "title": f"{org} {city_b} partnership notes",
            "text": (
                f"{org} signed a partnership with Tanaka Industries in {city_b} "
                f"to coordinate Asia-Pacific supply chains."
            ),
        },
    )
    assert upload_b.status_code == 200, upload_b.text
    doc_b_id = upload_b.json()["document_id"]

    # Single-entity filter: both docs surface (each mentions the org).
    r_single = client.get("/search", params={"q": org, "entity": org})
    assert r_single.status_code == 200, r_single.text
    single_doc_ids = {res["document_id"] for res in r_single.json()["results"]}
    assert doc_a_id in single_doc_ids, "?entity=<org> should surface doc A"
    assert doc_b_id in single_doc_ids, "?entity=<org> should surface doc B"

    # Two-entity AND filter: only doc A's chunk co-mentions <org> + city_a.
    r_and = client.get(
        "/search",
        params={"q": f"{org} {city_a}", "entity": [org, city_a]},
    )
    assert r_and.status_code == 200, r_and.text
    and_doc_ids = {res["document_id"] for res in r_and.json()["results"]}
    # Presence + absence together pin the feature down. A "doc B not present"
    # check alone passes vacuously when extraction silently misses city_a in
    # doc A — both docs absent satisfies the negative assertion.
    assert doc_a_id in and_doc_ids, (
        f"doc A should surface for ?entity={org}&entity={city_a}; "
        f"likely extraction missed {city_a}"
    )
    assert doc_b_id not in and_doc_ids, (
        f"doc B (no {city_a} mention) leaked into ?entity={org}&entity={city_a} results"
    )


@requires_voyage
@requires_anthropic
def test_entity_filter_unknown_name_returns_empty(
    client: TestClient,
    db_cleanup: None,
) -> None:
    """An ?entity= with no matching alias yields zero results under chunk-level AND."""
    r = client.get(
        "/search",
        params={"q": "anything", "entity": f"NonexistentEntity-{uuid4().hex}"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["results"] == [], "expected empty results for unknown entity name"


@requires_voyage
@requires_anthropic
def test_search_partial_resolve_yields_empty(
    client: TestClient,
    db_cleanup: None,
) -> None:
    """When some entity names resolve and others don't, the AND filter is
    unsatisfiable so /search returns empty — pinning the parse_entity_filter
    sentinel-padding behavior. Without that pad, the unresolved name silently
    drops and the filter relaxes to the resolved subset, returning chunks
    that mention only the resolved name (a UX trap: user filtered by X+Y,
    got chunks mentioning just X).
    """
    salt = uuid4().hex[:8]
    org = f"Pellican-{salt}"

    upload = client.post(
        "/text",
        json={
            "title": f"{org} background",
            "text": (
                f"{org} is a logistics startup based in Madrid. "
                f"Its CEO is Sofia Ruiz-{salt}."
            ),
        },
    )
    assert upload.status_code == 200, upload.text

    r = client.get(
        "/search",
        params={"q": org, "entity": [org, f"NonexistentEntity-{uuid4().hex}"]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["results"] == [], (
        "expected empty results: one resolvable + one unresolvable name "
        "should empty under chunk-level AND, not silently drop the unknown "
        "and return chunks for the rest"
    )


@requires_voyage
@requires_anthropic
def test_chat_falls_back_when_entity_filter_yields_zero(
    client: TestClient,
    db_cleanup: None,
) -> None:
    """The /chat fallback (DECISIONS.md §18.9) re-retrieves without the entity
    filter when the filter narrows candidates to zero. Constructs a corpus
    where two entities exist but no single chunk co-mentions them, then asks
    a question naming both. Without the fallback, retrieval would return
    empty, /chat would refuse, and `sources` would be empty.
    """
    salt = uuid4().hex[:8]
    org = f"Zorbnak-{salt}"
    location = f"Belgrade-{salt}"

    # Doc 1 — mentions org only.
    upload_org = client.post(
        "/text",
        json={
            "title": f"{org} earnings notes",
            "text": (
                f"{org} reported strong Q1 2026 earnings. "
                f"The company's revenue grew 30 percent year over year."
            ),
        },
    )
    assert upload_org.status_code == 200, upload_org.text

    # Doc 2 — mentions location only. No chunk in the corpus co-mentions both.
    upload_loc = client.post(
        "/text",
        json={
            "title": f"{location} city overview",
            "text": (
                f"{location} is a major European city with a population of two million. "
                f"Its central district hosts numerous cultural landmarks."
            ),
        },
    )
    assert upload_loc.status_code == 200, upload_loc.text

    # Question names both entities clearly so question-time extraction picks
    # them up. Filtered retrieval narrows to zero (no co-mention exists), so
    # the §18.9 fallback must re-run without the filter for sources to be
    # non-empty.
    r = client.post(
        "/chat",
        json={"question": f"What can you tell me about {org} and {location}?"},
    )
    assert r.status_code == 200, r.text

    body = r.json()
    assert body["sources"], (
        "expected non-empty sources; the §18.9 fallback should have re-run "
        "retrieval without the entity filter after the AND-filter narrowed "
        "candidates to zero (no chunk co-mentions both entities)"
    )
