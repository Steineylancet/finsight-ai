"""Department whitelist resolution — no network, no data."""

import pytest

from backend.graph.catalog import DEPARTMENTS, resolve_department


@pytest.mark.parametrize("raw,expected", [
    ("Software Engineering", "Software Engineering"),
    ("software engineering", "Software Engineering"),
    ("IT", "IT Infrastructure"),
    ("it infra", "IT Infrastructure"),
    ("HR", "Human Resources"),
    ("Data and Analytics", "Data & Analytics"),
    ("Sales EMEA", "Sales - EMEA"),
    ("marketing india", "Marketing - APAC"),
    ("Customer Success North America", "Customer Success - Americas"),
    ("Software Engneering", "Software Engineering"),  # typo
    ("Finance department", "Finance"),
])
def test_resolves_to_canonical(raw, expected):
    res = resolve_department(raw)
    assert res.ok and res.department == expected


@pytest.mark.parametrize("raw,group", [
    ("Marketing", "Marketing"),
    ("Sales", "Sales"),
    ("customer success", "Customer Success"),
])
def test_regional_group_without_region_is_ambiguous(raw, group):
    res = resolve_department(raw)
    assert not res.ok
    assert len(res.candidates) == 3
    assert all(c.startswith(group + " - ") for c in res.candidates)
    assert "did you mean" in res.message(raw)


@pytest.mark.parametrize("raw", ["Innovation", "Blockchain Lab", "", None])
def test_unknown_department_is_rejected(raw):
    res = resolve_department(raw)
    assert not res.ok


def test_unknown_message_lists_valid_departments():
    msg = resolve_department("Innovation").message("Innovation")
    assert "not a department" in msg and "Software Engineering" in msg


def test_every_canonical_name_resolves_to_itself():
    for d in DEPARTMENTS:
        assert resolve_department(d).department == d
