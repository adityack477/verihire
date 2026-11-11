# tests/test_skills.py
import json
import re

import pytest

from ml.docint.ingest import skills
from ml.docint.ingest.skills import SKILLS_DICT_PATH, extract_skills, load_skills_dict


def _reject_dupes(pairs):
    keys = [k for k, _ in pairs]
    assert len(keys) == len(set(keys)), "duplicate keys in skills_dict.json"
    return dict(pairs)


def _norm(term):
    return re.sub(r"[\s\-]+", "", term.lower())


def test_dictionary_size_lowercase_and_no_duplicates():
    json.loads(SKILLS_DICT_PATH.read_text(encoding="utf-8"), object_pairs_hook=_reject_dupes)
    data = load_skills_dict()
    assert len(data) >= 300
    owner = {}
    for canon, aliases in data.items():
        assert canon == canon.lower() and canon.strip() == canon
        for term in [canon, *aliases]:
            assert term == term.lower()
            assert owner.setdefault(_norm(term), canon) == canon


def test_java_vs_javascript():
    assert extract_skills("Java") == ["java"]
    assert extract_skills("JavaScript") == ["javascript"]
    assert extract_skills("Java, JavaScript") == ["java", "javascript"]


def test_aliases():
    assert extract_skills("JS") == ["javascript"]
    assert extract_skills("Docker, K8s") == ["docker", "kubernetes"]
    assert extract_skills("scikit-learn, sklearn") == ["scikit-learn"]
    assert extract_skills("Machine-Learning and NLP") == ["machine learning", "natural language processing"]


def test_punctuation_heavy_terms():
    assert extract_skills("C++, C#") == ["c++", "c#"]
    assert extract_skills("C/C++") == ["c", "c++"]
    assert extract_skills("C#, .NET") == ["c#", ".net"]
    assert extract_skills("Node.js and Express") == ["node.js", "express.js"]
    assert extract_skills("JS, Node.js") == ["javascript", "node.js"]
    assert extract_skills("Vue.js") == ["vue.js"]


def test_sql_family_not_confused():
    assert extract_skills("MySQL, PostgreSQL, NoSQL, SQL") == ["mysql", "postgresql", "nosql", "sql"]


def test_go_and_short_token_false_positives():
    assert extract_skills("I will go home and go to the market") == []
    assert extract_skills("Go to market strategy") == []
    assert extract_skills("Languages: Python, Go, SQL") == ["python", "go", "sql"]
    assert extract_skills("Golang services") == ["go"]
    assert extract_skills("R, Python, R&D") == ["r", "python"]
    assert extract_skills("we react to feedback and excel at teamwork") == []


def test_go_in_prose_regression_g23():
    assert extract_skills("Built services in Go.") == ["go"]
    assert extract_skills("Wrote services using Go, Docker") == ["go", "docker"]
    assert extract_skills("Experience with Go and Python") == ["go", "python"]


def test_go_prose_false_positives_still_rejected():
    assert extract_skills("Go to market strategy") == []
    assert extract_skills("Let's work in Go to market teams") == []
    assert extract_skills("Go and see the market") == []
    assert extract_skills("Ready to go in the morning. Then go home.") == []
    assert extract_skills("We will go with the plan") == []


def test_duplicates_and_determinism():
    text = "Python python PYTHON python3, SQL, sql"
    assert extract_skills(text) == ["python", "sql"]
    assert extract_skills(text) == extract_skills(text)


def test_cwd_independent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skills._get_index.cache_clear()
    assert extract_skills("Python") == ["python"]


def test_non_str_input_raises():
    with pytest.raises(TypeError):
        extract_skills(123)  # type: ignore[arg-type]
