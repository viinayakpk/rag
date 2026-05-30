"""Hand-crafted corpus + golden questions for the graph-vs-baseline eval.

Fifteen short text documents — five "target" docs that carry the answers
to each golden question, plus ten "distractor" docs that share entity
surface forms (Quintara, Garcia, Berlin, Tokyo, Tanaka, TechSummit)
with the targets but on different topics. The corpus is sized
deliberately above `CHAT_TOP_K = 8` so baseline retrieval is forced to
choose 8 of 15 (real lexical/semantic competition) rather than dump
everything wholesale, which is what happens when the corpus fits inside
top_k. Each co-mention question is constructed so exactly one document
satisfies the AND filter — the brief's "X together with Y" shape.

All document titles use the EVAL/ prefix so the runner can clean up its
own data without touching other ingested content. Each question carries
explicit pass criteria: case-insensitive substrings the answer must
contain, and document titles that must appear in the response sources.
"""

from typing import TypedDict


class Document(TypedDict):
    title: str
    text: str


class Question(TypedDict):
    question: str
    expected_substrings: list[str]
    expected_document_titles: list[str]


CORPUS: list[Document] = [
    # === Target documents — each carries the answer to one or more questions. ===
    {
        "title": "EVAL/Quintara Berlin office",
        "text": (
            "Quintara Industries opened its Berlin headquarters on March 15, 2026. "
            "CEO Maria Garcia announced that the Berlin office would lead European "
            "operations and employ 200 engineers."
        ),
    },
    {
        "title": "EVAL/Quintara Tokyo partnership",
        "text": (
            "Quintara Industries signed a partnership with Tanaka Holdings in Tokyo "
            "to coordinate Asia-Pacific supply chains. The agreement spans five years "
            "and covers twelve markets."
        ),
    },
    {
        "title": "EVAL/Garcia TechSummit keynote",
        "text": (
            "Maria Garcia delivered the opening keynote at TechSummit 2026 in San "
            "Francisco on April 8, 2026. Her talk focused on autonomous logistics "
            "platforms and predictive routing."
        ),
    },
    {
        "title": "EVAL/Quintara Q1 earnings",
        "text": (
            "Quintara Industries reported Q1 2026 revenue of $4.2 billion, a 23% "
            "year-over-year increase. The company attributed growth to new "
            "infrastructure contracts."
        ),
    },
    {
        "title": "EVAL/Tanaka Holdings background",
        "text": (
            "Tanaka Holdings, founded in 1962 by Hiroshi Tanaka, operates 47 "
            "manufacturing facilities across Japan. The company is headquartered "
            "in Tokyo."
        ),
    },
    # === Distractor documents — share entity surface forms with the targets ===
    # === but on different topics. None of them satisfy any question's       ===
    # === AND filter (verified by hand against the question expectations).   ===
    {
        # Quintara mentioned, but no Berlin / Tokyo / Garcia / Tanaka / Q1 2026.
        "title": "EVAL/Quintara CFO appointment",
        "text": (
            "Quintara Industries appointed Patricia Yang as Chief Financial Officer "
            "effective February 2026. Yang previously served as VP of Finance at "
            "Helix Aerospace and brings fifteen years of capital-markets experience "
            "to the role."
        ),
    },
    {
        # Quintara mentioned, but the location is Boston (not Berlin / Tokyo).
        "title": "EVAL/Quintara Boston research lab",
        "text": (
            "Quintara Industries opened a new artificial-intelligence research lab "
            "in Boston in March 2026. The lab will focus on autonomous logistics "
            "and employs eighty researchers led by director Wei Chen."
        ),
    },
    {
        # Quintara mentioned, no locations or people.
        "title": "EVAL/Quintara stock split",
        "text": (
            "Quintara Industries announced a three-for-one stock split in April 2026, "
            "citing strong investor interest and a goal of broader retail accessibility. "
            "The split takes effect on May 15."
        ),
    },
    {
        # Garcia + Quintara, but not Berlin / TechSummit — narrows for Q3 / Q4.
        "title": "EVAL/Maria Garcia profile",
        "text": (
            "Maria Garcia, Quintara Industries CEO since 2023, was profiled in Wired "
            "magazine's April 2026 leadership issue. The profile traces her career "
            "from her early days at logistics startup Routelane to her current role."
        ),
    },
    {
        # Berlin, no Quintara / Garcia.
        "title": "EVAL/Berlin tech scene",
        "text": (
            "Berlin's startup ecosystem grew eighteen percent in 2025, with new "
            "venture investment topping four billion euros. The Mitte district hosts "
            "two hundred of the city's tech firms."
        ),
    },
    {
        # Tokyo, no Quintara / Tanaka.
        "title": "EVAL/Tokyo real estate",
        "text": (
            "Tokyo office rents climbed six percent in early 2026 as tech firms "
            "expanded their footprints. Premium districts including Marunouchi and "
            "Roppongi led the increase."
        ),
    },
    {
        # Tanaka Holdings, but Osaka not Tokyo, no Quintara.
        "title": "EVAL/Tanaka Osaka expansion",
        "text": (
            "Tanaka Holdings opened a fifth manufacturing facility in Osaka in "
            "February 2026. The plant will produce industrial sensors and employ "
            "one hundred fifty workers."
        ),
    },
    {
        # TechSummit + San Francisco, no Garcia.
        "title": "EVAL/TechSummit sponsors",
        "text": (
            "TechSummit 2026 announced its full sponsor list, with NeoChip and "
            "Atlas Capital headlining as platinum sponsors. The conference runs "
            "April 7-10 in San Francisco."
        ),
    },
    {
        # Pure topic doc — competes lexically with Q2/Q6 (logistics, Europe).
        "title": "EVAL/European logistics market",
        "text": (
            "Europe's logistics-technology market is projected to reach two hundred "
            "eighty billion euros by 2028, driven by automation, last-mile delivery "
            "innovation, and supply-chain digitization. Industry analysts cite labor "
            "shortages as a key driver of investment."
        ),
    },
    {
        # Pure topic doc — competes lexically with Q2/Q6 (Asia-Pacific supply chain).
        "title": "EVAL/Asia-Pacific supply chain",
        "text": (
            "Asia-Pacific supply chains face increasing pressure from currency "
            "volatility and shifting trade policies. Manufacturers are diversifying "
            "suppliers across Vietnam, Indonesia, and Thailand to reduce concentration risk."
        ),
    },
]

QUESTIONS: list[Question] = [
    # Single-entity sanity — both modes should pass.
    {
        "question": "What did Quintara Industries report in Q1 2026?",
        "expected_substrings": ["4.2 billion"],
        "expected_document_titles": ["EVAL/Quintara Q1 earnings"],
    },
    # Co-mention narrowing: Quintara appears in 3 docs (Berlin, Tokyo, earnings);
    # Berlin appears only in doc 1. Graph-on AND filter narrows to that one.
    {
        "question": "What did Quintara Industries announce in Berlin?",
        "expected_substrings": ["headquarters"],
        "expected_document_titles": ["EVAL/Quintara Berlin office"],
    },
    # Garcia + Berlin AND filter — Garcia is in doc 1 (Berlin) and doc 3
    # (TechSummit/SF); only doc 1 mentions both.
    {
        "question": "What did Maria Garcia announce in Berlin?",
        "expected_substrings": ["200 engineers"],
        "expected_document_titles": ["EVAL/Quintara Berlin office"],
    },
    # Garcia + TechSummit AND filter — only doc 3.
    {
        "question": "What did Maria Garcia present at TechSummit 2026?",
        "expected_substrings": ["autonomous logistics"],
        "expected_document_titles": ["EVAL/Garcia TechSummit keynote"],
    },
    # Single-entity sanity — both modes should pass.
    {
        "question": "Where is Tanaka Holdings headquartered?",
        "expected_substrings": ["Tokyo"],
        "expected_document_titles": ["EVAL/Tanaka Holdings background"],
    },
    # Quintara + Tanaka AND filter — Tanaka is in docs 2 and 5; only doc 2
    # mentions both, and only doc 2 has the partnership detail.
    {
        "question": (
            "What is the size of the partnership between Quintara Industries "
            "and Tanaka Holdings?"
        ),
        "expected_substrings": ["five years"],
        "expected_document_titles": ["EVAL/Quintara Tokyo partnership"],
    },
    # Single-entity sanity — both modes should pass.
    {
        "question": "Who founded Tanaka Holdings?",
        "expected_substrings": ["Hiroshi Tanaka"],
        "expected_document_titles": ["EVAL/Tanaka Holdings background"],
    },
]
