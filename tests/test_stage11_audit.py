import pathlib
import pytest
from evacuation_simulation.simulation.rules import load_regulatory_config

def test_unverified_rule_is_marked_unverified():
    cfg = load_regulatory_config()
    rules = cfg.get("rules", {})
    r3 = rules.get("R3_EXIT_SEPARATION", {})
    assert r3.get("non_sprinklered", {}).get("verified") is False
    assert r3.get("sprinklered", {}).get("verified") is False

def test_regulatory_metadata_contains_clause():
    cfg = load_regulatory_config()
    rules = cfg.get("rules", {})
    r4 = rules.get("R4_EXIT_WIDTH", {})
    clause = r4.get("absolute_minimum_width_per_exit", {}).get("clause", "")
    assert "Table 4" in clause or "NBC 2016" in clause

def test_conditional_rule_does_not_become_universal():
    cfg = load_regulatory_config()
    rules = cfg.get("rules", {})
    r2 = rules.get("R2_MINIMUM_EXIT_COUNT", {})
    assert r2.get("min_exits_absolute", {}).get("verified") is False
    assert "conditionally verified" in str(r2.get("min_exits_absolute", {}).get("clause")).lower()

def test_width_value_is_separate_from_physical_width():
    cfg = load_regulatory_config()
    rules = cfg.get("rules", {})
    r4 = rules.get("R4_EXIT_WIDTH", {})
    assert r4.get("width_per_person_m", {}).get("value") is None
    assert r4.get("width_per_person_m", {}).get("verified") is False

