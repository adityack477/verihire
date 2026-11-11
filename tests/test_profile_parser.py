# tests/test_profile_parser.py
from datetime import date

import pytest

from ml.docint.ingest.chunker import Chunk, chunk_document
from ml.docint.ingest.pii_scrubber import scrub_pages
from ml.docint.ingest.profile_parser import PARSER_VERSION, ResumeProfile, parse_profile

AS_OF = date(2026, 1, 1)


def mk(section, text, idx=0):
    return Chunk(chunk_index=idx, section=section, page_no=1, char_start=0, char_end=len(text),
                 text_original=text, text_scrubbed=text, token_count=len(text.split()))


def years(text, as_of=AS_OF):
    return parse_profile([mk("experience", text)], as_of=as_of).years_experience


def test_overlapping_dates_are_merged():
    assert years("Data Analyst, Acme\nJan 2020 - Dec 2021\n\nDeveloper, Beta\nJun 2021 - Jun 2022") == 2.5


def test_non_overlapping_dates_sum():
    assert years("Jan 2020 - Dec 2020\nJan 2022 - Dec 2022") == 2.0


def test_present_and_current():
    assert years("Mar 2024 - Present", as_of=date(2025, 3, 15)) == 1.1
    assert years("Jun 2025 - current", as_of=date(2025, 12, 31)) == 0.6


def test_other_date_formats():
    assert years("2021 - 2023") == 2.0
    assert years("01/2022 - 12/2022") == 1.0
    assert years("May - June 2026", as_of=date(2026, 11, 10)) == 0.2


def test_end_clamped_and_future_ignored():
    assert years("Jan 2025 - Dec 2030") == 1.1
    assert years("Jan 2030 - Present") is None


def test_malformed_dates_ignored():
    text = "Foo 2020 - Bar 2021\n13/2020 - 05/2021\nMar 2022 - Jan 2021\nJan 2024 - Foo 2025"
    assert years(text) is None


def test_no_experience_inferred_from_education_dates():
    profile = parse_profile([mk("education", "B.Tech, KIIT University\n2023 - 2027")], as_of=AS_OF)
    assert profile.years_experience is None


@pytest.mark.parametrize("text,level", [
    ("Diploma in Electronics, Govt Polytechnic", 1),
    ("B.Tech in Computer Science", 2),
    ("B.E. in Mechanical Engineering", 2),
    ("BSc Physics", 2),
    ("Bachelor of Science in Statistics", 2),
    ("M.Tech in AI", 3),
    ("MSc Data Science", 3),
    ("MBA, XYZ Business School", 3),
    ("Ph.D. in Physics", 4),
    ("B.Tech 2019; M.Tech 2021", 3),
    ("Higher Secondary, CBSE 94%", 0),
])
def test_education_mapping(text, level):
    assert parse_profile([mk("education", text)], as_of=AS_OF).education_level == level


@pytest.mark.parametrize("text", ["BS in Computer Science, Cedar University", "B.S. in Physics",
                                   "BS (Computer Science)"])
def test_bs_degree_is_bachelor_regression_g23(text):
    assert parse_profile([mk("education", text)], as_of=AS_OF).education_level == 2


@pytest.mark.parametrize("text", ["jobs in demand", "Higher Secondary, CBS 94%", "bs in practice"])
def test_bs_pattern_has_no_false_positives(text):
    assert parse_profile([mk("education", text)], as_of=AS_OF).education_level == 0


def test_education_fields():
    p = parse_profile([mk("education", "B.Tech in Computer Science & Engineering, KIIT")], as_of=AS_OF)
    assert p.education_fields == ["computer science"]
    p = parse_profile([mk("education", "MSc Statistics and Data Science")], as_of=AS_OF)
    assert p.education_fields == ["statistics", "data science"]


def test_titles():
    text = ("EXPERIENCE\n\nData Analyst Intern, Acme Analytics\nJan 2024 - Mar 2024\n"
            "\u2022 Led a team of engineers on dashboards\n\nSoftware Engineer at Beta Corp\n2022 - 2023\n"
            "Intern | DRDO CAIR (Centre for Artificial Intelligence & Robotics)")
    titles = parse_profile([mk("experience", text)], as_of=AS_OF).titles
    assert titles == ["Data Analyst Intern", "Software Engineer", "Intern"]


# tests/test_profile_parser.py: replace the failing assertion in test_skills_and_certifications
def test_skills_and_certifications():
    p = parse_profile([mk("skills", "Python, SQL, scikit-learn"),
                       mk("certifications", "CERTIFICATIONS & COURSES\nGoogle Data Analytics Certificate\n"
                                            "\u2022 AWS Cloud Practitioner\n2024", 1)], as_of=AS_OF)
    assert p.skills == ["python", "sql", "scikit-learn", "data analysis", "aws"]
    assert p.certifications == ["Google Data Analytics Certificate", "AWS Cloud Practitioner"]

def test_missing_data_gives_defaults():
    p = parse_profile([], as_of=AS_OF)
    assert p == ResumeProfile()
    assert p.skills == [] and p.titles == [] and p.years_experience is None
    assert p.education_level == 0 and p.summary is None and p.parser_version == PARSER_VERSION


def test_summary_is_pii_safe():
    text = ("Aarav Mehta\naarav.mehta@example.com | +91 98765 43210\n\nSUMMARY\n"
            "Aarav Mehta is a data analyst. Reach him at aarav.mehta@example.com or +91 98765 43210.\n\n"
            "SKILLS\nPython, SQL")
    pages = [{"page_no": 1, "text": text}]
    chunks = chunk_document(pages, scrub_pages(pages, use_spacy=False).spans)
    summary = parse_profile(chunks, as_of=AS_OF).summary
    assert summary and "[NAME]" in summary
    assert "Aarav" not in summary and "@" not in summary and "98765" not in summary
    assert "SUMMARY" not in summary


def test_summary_length_cap():
    long = "SUMMARY\n" + " ".join(["Analytical"] * 100)
    summary = parse_profile([mk("summary", long)], as_of=AS_OF).summary
    assert summary and len(summary) <= 303
