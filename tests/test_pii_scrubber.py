# tests/test_pii_scrubber.py
import dataclasses

from ml.docint.ingest.pii_scrubber import PiiEntities, scrub_pages, scrub_text


def run(text):
    result = scrub_pages([{"page_no": 1, "text": text}], use_spacy=False)
    return result, scrub_text(text, result.spans[1])


def test_entities_fields_match_contract():
    names = {f.name for f in dataclasses.fields(PiiEntities)}
    assert names == {"full_name", "emails", "phones", "urls", "other"}


def test_emails():
    text = "Contact: aarav.mehta@example.com or a_b+tag@sub.domain.co.in"
    result, scrubbed = run(text)
    assert result.entities.emails == ["aarav.mehta@example.com", "a_b+tag@sub.domain.co.in"]
    assert scrubbed.count("[EMAIL]") == 2


def test_indian_phones():
    text = "Skills\n9876543210\n+91-98765-43210\n098765 43210\n+91 98765 43210"
    result, scrubbed = run(text)
    assert len(result.entities.phones) == 4
    assert "[PHONE]" in scrubbed
    for raw in result.entities.phones:
        assert raw not in scrubbed


def test_international_phones():
    text = "Phone: +1 (415) 555-0132\nAlt: +44 20 7946 0958\nUS: 415-555-0132"
    result, scrubbed = run(text)
    assert "+1 (415) 555-0132" in result.entities.phones
    assert "+44 20 7946 0958" in result.entities.phones
    assert "415-555-0132" in result.entities.phones
    assert "555-0132" not in scrubbed


def test_urls_linkedin_github():
    text = ("https://www.linkedin.com/in/aarav-mehta-123\n"
            "See github.com/aarav-mehta.\nSite: www.aarav.dev")
    result, scrubbed = run(text)
    assert "https://www.linkedin.com/in/aarav-mehta-123" in result.entities.urls
    assert "github.com/aarav-mehta" in result.entities.urls
    assert "www.aarav.dev" in result.entities.urls
    assert "linkedin.com" not in scrubbed and "github.com" not in scrubbed
    assert "[URL]." in scrubbed


def test_name_from_top_of_page_and_other_occurrences():
    text = "Aarav Mehta\naarav.mehta@example.com\n\nSUMMARY\nReference: Aarav Mehta, Pune."
    result, scrubbed = run(text)
    assert result.entities.full_name == "Aarav Mehta"
    assert "Aarav Mehta" not in scrubbed
    assert scrubbed.count("[NAME]") == 2


def test_name_from_email_local_part():
    text = "RESUME\nPage 1 of 2\nUpdated 2026\nConfidential\nPriya Nair\npriya.nair@example.com\nSKILLS\nPython"
    result, _ = run(text)
    assert result.entities.full_name == "Priya Nair"


def test_no_name_for_job_title_first_line():
    result, _ = run("Machine Learning Engineer\nSkills\nPython, Java")
    assert result.entities.full_name is None


def test_false_positives_preserved():
    text = ("Aarav Mehta\n"
            "Python Developer at Acme Analytics Pvt Ltd, 2019 - 2023. CGPA 8.75. "
            "Handled 1000000 records; 12th: 94.2%. Node.js ASP.NET Vue.js TensorFlow\n"
            "KIIT University, Bhubaneswar")
    result, scrubbed = run(text)
    assert result.entities.phones == []
    assert result.entities.urls == []
    assert result.entities.emails == []
    for keep in ["Acme Analytics Pvt Ltd", "KIIT University", "Python", "TensorFlow", "Node.js", "2019 - 2023"]:
        assert keep in scrubbed


def test_all_placeholders_and_no_raw_values():
    text = ("Aarav Mehta\naarav.mehta@example.com | +91 98765 43210\n"
            "linkedin.com/in/aarav-mehta\nAddress: 12 MG Road, Pune")
    result, scrubbed = run(text)
    for placeholder in ["[NAME]", "[EMAIL]", "[PHONE]", "[URL]", "[ADDRESS]"]:
        assert placeholder in scrubbed
    raw_values = (result.entities.emails + result.entities.phones + result.entities.urls
                  + result.entities.other + [result.entities.full_name])
    for raw in raw_values:
        assert raw not in scrubbed
    assert "12 MG Road, Pune" in result.entities.other