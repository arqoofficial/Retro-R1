import json
import os

import pytest

from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun, rdchiralRunText

_CASES_PATH = os.path.join(os.path.dirname(__file__), "test_rdchiral_cases.json")
with open(_CASES_PATH, encoding="utf-8") as fid:
    _TEST_CASES = json.load(fid)


@pytest.mark.parametrize(
    "case",
    _TEST_CASES,
    ids=[f"case_{i}" for i in range(len(_TEST_CASES))],
)
def test_rdchiral_run_text(case):
    assert rdchiralRunText(case["smarts"], case["smiles"]) == case["expected"]


@pytest.mark.parametrize(
    "case",
    _TEST_CASES,
    ids=[f"case_{i}" for i in range(len(_TEST_CASES))],
)
def test_rdchiral_run_init(case):
    rxn = rdchiralReaction(case["smarts"])
    reactants = rdchiralReactants(case["smiles"])
    assert all(rdchiralRun(rxn, reactants) == case["expected"] for _ in range(3))
