# tests/test_chunker.py
from ml.docint.ingest.chunker import HARD_MAX, Chunk, chunk_document
from ml.docint.ingest.pii_scrubber import scrub_pages

RESUME = """Aarav Mehta
aarav.mehta@example.com | +91 98765 43210

PROFESSIONAL SUMMARY
Data analyst focused on dashboards and forecasting.

WORK EXPERIENCE
Data Analyst Intern, Acme Analytics
Jan 2024 - Mar 2024
- Built weekly reporting dashboards for operations.
- Cleaned and validated sales data.

TECHNICAL SKILLS
Python, SQL, Pandas

EDUCATION
B.Tech in Computer Science, KIIT University
2023 - 2027

ACADEMIC PROJECTS
Churn Predictor
- Trained a gradient boosting model.

CERTIFICATIONS & COURSES
Google Data Analytics Certificate
"""


def pages(text):
    return [{"page_no": 1, "text": text}]


def build(text):
    pg = pages(text)
    scrub = scrub_pages(pg, use_spacy=False)
    return pg, scrub, chunk_document(pg, scrub.spans)


def test_section_detection():
    _, _, chunks = build(RESUME)
    sections = {c.section for c in chunks}
    assert {"header", "summary", "experience", "skills", "education", "projects",
            "certifications"} <= sections


def test_exact_offsets_and_scrubbing():
    pg, _, chunks = build(RESUME)
    text = pg[0]["text"]
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert isinstance(c, Chunk)
        assert 0 <= c.char_start < c.char_end <= len(text)
        assert c.text_original == text[c.char_start:c.char_end]
        assert c.text_scrubbed.strip()
        assert c.embedding_text() == c.text_scrubbed
        assert c.token_count > 0
        assert "aarav.mehta@example.com" not in c.text_scrubbed
    header = chunks[0]
    assert "aarav.mehta@example.com" in header.text_original
    assert "[EMAIL]" in header.text_scrubbed


def test_role_header_kept_with_first_bullet():
    _, _, chunks = build(RESUME)
    exp = [c for c in chunks if c.section == "experience"][0]
    assert "Data Analyst Intern" in exp.text_original
    assert "Built weekly reporting dashboards" in exp.text_original


def test_target_and_max_sizes_with_bullet_boundaries():
    bullets = "\n".join(
        f"- Delivered analysis number {i} improving reporting accuracy and turnaround for stakeholders across teams."
        for i in range(30))
    text = "EXPERIENCE\n\nData Analyst Intern, Acme Analytics\nJan 2024 - Mar 2024\n\n" + bullets
    _, _, chunks = build(text)
    assert len(chunks) >= 3
    assert all(c.section == "experience" for c in chunks)
    assert all(len(c.text_original) <= HARD_MAX and len(c.text_scrubbed) <= HARD_MAX for c in chunks)
    assert chunks[0].text_original.startswith("EXPERIENCE")
    for c in chunks[1:]:
        assert c.text_original.startswith("- ")
    for c in chunks[:-1]:
        assert 800 <= len(c.text_original) <= 1200


def test_long_single_line_is_split_under_hard_max():
    text = " ".join(["word"] * 600)
    _, _, chunks = build(text)
    assert len(chunks) >= 3
    assert all(len(c.text_original) <= HARD_MAX for c in chunks)
    assert all(c.text_original == text[c.char_start:c.char_end] for c in chunks)
    assert sum(len(c.text_original.split()) for c in chunks) == 600


def test_empty_and_short_resumes():
    assert chunk_document(pages(""), {1: []}) == []
    assert chunk_document(pages("  \n\n \n"), {1: []}) == []
    _, _, chunks = build("Aarav Mehta\nPython developer")
    assert len(chunks) == 1
    assert chunks[0].text_original == "Aarav Mehta\nPython developer"


def test_multi_page_chunks_reference_their_page():
    pg = [{"page_no": 1, "text": "SKILLS\nPython, SQL"}, {"page_no": 2, "text": "EDUCATION\nB.Tech, KIIT University"}]
    scrub = scrub_pages(pg, use_spacy=False)
    chunks = chunk_document(pg, scrub.spans)
    assert [(c.page_no, c.section) for c in chunks] == [(1, "skills"), (2, "education")]
    for c in chunks:
        assert c.text_original == pg[c.page_no - 1]["text"][c.char_start:c.char_end]