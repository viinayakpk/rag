"""Unit tests for graph_resolve helpers.

`_anchor_existing_canonicals` is the post-Sonnet anchoring step that prevents
the resolver from renaming a name that's already canonical in the DB —
otherwise persist_graph would write a duplicate `entities` row for the same
real-world entity. These tests exercise the helper directly with hand-built
Cluster inputs (no LLM, no DB) so the anchor's behavior is locked down
deterministically.
"""

from app.graph_resolve import _anchor_existing_canonicals
from app.models import Cluster


def test_anchor_rewrites_when_sonnet_renames_existing_canonical() -> None:
    """The bug case: Sonnet picked a new canonical, but an alias is already in the DB."""
    clusters = [
        Cluster(canonical="Acme Corporation", aliases=["Acme", "Acme Corporation"])
    ]
    result = _anchor_existing_canonicals(clusters, existing_canonicals={"Acme"})

    assert result[0].canonical == "Acme"
    assert set(result[0].aliases) == {"Acme", "Acme Corporation"}


def test_anchor_noop_when_sonnet_kept_existing_canonical() -> None:
    """Sonnet already chose the existing canonical — leave it alone."""
    clusters = [
        Cluster(canonical="Acme", aliases=["Acme", "Acme Inc."])
    ]
    result = _anchor_existing_canonicals(clusters, existing_canonicals={"Acme"})

    assert result[0].canonical == "Acme"


def test_anchor_noop_when_no_alias_overlaps_existing() -> None:
    """A purely new entity with no overlap with existing canonicals — keep Sonnet's choice."""
    clusters = [
        Cluster(canonical="Globex Corp", aliases=["Globex", "Globex Corp"])
    ]
    result = _anchor_existing_canonicals(clusters, existing_canonicals={"Acme"})

    assert result[0].canonical == "Globex Corp"


def test_anchor_noop_when_existing_set_empty() -> None:
    """First ingest of a type — there are no canonicals to anchor against."""
    clusters = [
        Cluster(canonical="Acme Corp", aliases=["Acme", "Acme Corp"])
    ]
    result = _anchor_existing_canonicals(clusters, existing_canonicals=set())

    assert result[0].canonical == "Acme Corp"


def test_anchor_picks_one_when_multiple_existing_in_aliases() -> None:
    """Cluster merges two pre-existing canonicals — pick one as the survivor."""
    clusters = [
        Cluster(
            canonical="Acme Worldwide",
            aliases=["Acme Corporation", "Acme Inc.", "Acme Worldwide"],
        )
    ]
    result = _anchor_existing_canonicals(
        clusters,
        existing_canonicals={"Acme Corporation", "Acme Inc."},
    )

    assert result[0].canonical in {"Acme Corporation", "Acme Inc."}


def test_anchor_handles_multiple_clusters_independently() -> None:
    """Each cluster is anchored on its own merits — overlap in one doesn't bleed into another."""
    clusters = [
        Cluster(canonical="Acme Corp", aliases=["Acme", "Acme Corp"]),
        Cluster(canonical="Globex Inc.", aliases=["Globex", "Globex Inc."]),
    ]
    result = _anchor_existing_canonicals(clusters, existing_canonicals={"Acme"})

    assert result[0].canonical == "Acme"
    assert result[1].canonical == "Globex Inc."


def test_anchor_returns_empty_for_empty_input() -> None:
    """No clusters → empty list back, no errors."""
    assert _anchor_existing_canonicals([], existing_canonicals={"Acme"}) == []
