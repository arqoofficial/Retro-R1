#!/usr/bin/env python3
"""
Experiments — single-step MLP on CPU (Mac-friendly), no LLM.

Compares Retro*-style one-step predictions against ground-truth first steps from
routes_possible_test_hard.pkl:

  - top-k = 5 / 10 / 20 (same as agent config tool.topk)
  - checkpoint V1 vs V2 (and optional V3)
  - fraction of precursors marked available vs unavailable
  - hit@k: predicted reactant set matches the first retrosynthetic step in the route

Run from the repository root:

  uv run python scripts/experiments_mlp_one_step.py
  uv run python scripts/experiments_mlp_one_step.py --checkpoints v1,v2 --topk 5,10,20 -n 10
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = REPO_ROOT / "one_step_model" / "template_rules_1.dat"
DEFAULT_TEST_PKL = REPO_ROOT / "dataset" / "routes_possible_test_hard.pkl"

CHECKPOINTS = {
    "v1": REPO_ROOT / "one_step_model" / "saved_rollout_state_1_2048.ckpt",
    "v2": REPO_ROOT / "one_step_model" / "retro_star_value_ours.ckpt",
    "v3": REPO_ROOT / "one_step_model" / "retro_star_zero_ours.ckpt",
}

# Same as SingleStepRetroTool: V1 calls MLP with topk=50 internally, then truncates.
V1_INTERNAL_TOPK = 50


def _add_repo_to_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def canonicalize_smiles_clear_map(smiles: str) -> str:
    """Match examples/data_preprocess/reaction.py and tool_env_retro."""
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        return ""
    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return ""


def reactants_key(smiles: str) -> tuple[str, ...]:
    """Sorted canonical fragments — multiset comparison for one-step match."""
    parts = []
    for frag in smiles.split("."):
        frag = frag.strip()
        if not frag:
            continue
        canon = canonicalize_smiles_clear_map(frag)
        if canon:
            parts.append(canon)
    return tuple(sorted(parts))


def extract_first_step_ground_truth(route: list[str]) -> tuple[str, str, tuple[str, ...]]:
    """First list entry: product>>reactants (retro direction)."""
    if ">>" not in route[0]:
        raise ValueError(f"Invalid route step: {route[0]!r}")
    target, gt_reactants = route[0].split(">>", 1)
    return target.strip(), gt_reactants.strip(), reactants_key(gt_reactants)


def prediction_matches_gt(predicted_reactants: str, gt_key: tuple[str, ...]) -> bool:
    return reactants_key(predicted_reactants) == gt_key


def find_gt_hit(
    predictions: list[str], gt_key: tuple[str, ...], topk: int
) -> dict:
    for rank, pred in enumerate(predictions[:topk]):
        if prediction_matches_gt(pred, gt_key):
            return {"hit": True, "hit_rank": rank, "hit_reactants": pred}
    return {"hit": False, "hit_rank": None, "hit_reactants": None}


def load_starting_mols(path: Path, cache_path: Path | None) -> set[str]:
    if cache_path and cache_path.is_file():
        with cache_path.open("rb") as f:
            return pickle.load(f)

    if not path.is_file():
        raise FileNotFoundError(
            f"Starting molecules file not found: {path}\n"
            "Download retro_data.zip (see README.md) and extract dataset/origin_dict.csv."
        )

    starting = set(pd.read_csv(path, usecols=["mol"])["mol"])

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(starting, f, protocol=pickle.HIGHEST_PROTOCOL)

    return starting


@dataclass
class TestCase:
    route_index: int
    target_smiles: str
    gt_reactants: str | None
    gt_reactants_key: tuple[str, ...] | None


def load_test_cases(
    test_pkl: Path,
    indices: Iterable[int] | None,
    n_mols: int,
    smiles_list: list[str] | None,
) -> list[TestCase]:
    if smiles_list:
        return [
            TestCase(i, s, None, None) for i, s in enumerate(smiles_list)
        ]

    with test_pkl.open("rb") as f:
        routes = pickle.load(f)

    if indices is not None:
        idx_list = list(indices)
    else:
        idx_list = list(range(min(n_mols, len(routes))))

    cases: list[TestCase] = []
    for i in idx_list:
        target, gt_smi, gt_key = extract_first_step_ground_truth(routes[i])
        cases.append(
            TestCase(
                route_index=i,
                target_smiles=target,
                gt_reactants=gt_smi,
                gt_reactants_key=gt_key,
            )
        )
    return cases


def mark_availability(reactants_smiles: str, starting_mols: set[str]) -> list[dict]:
    parts = []
    for smi in reactants_smiles.split("."):
        available = smi in starting_mols
        parts.append(
            {
                "smiles": smi,
                "status": "available" if available else "unavailable",
            }
        )
    return parts


@dataclass
class ReactionSummary:
    rank: int
    score: float
    score_reference: float
    reactants: str
    matches_ground_truth: bool
    precursors: list[dict]
    n_available: int
    n_unavailable: int
    all_available: bool
    any_available: bool


@dataclass
class MoleculeRun:
    route_index: int
    target_smiles: str
    checkpoint: str
    topk: int
    n_predictions: int
    ground_truth_reactants: str | None
    hit: bool
    hit_rank: int | None
    reactions: list[ReactionSummary]
    frac_precursors_available: float
    frac_reactions_all_available: float


def run_one_molecule(
    model,
    case: TestCase,
    checkpoint_key: str,
    topk: int,
    starting_mols: set[str],
) -> MoleculeRun | None:
    is_v1 = checkpoint_key == "v1"
    mlp_topk = V1_INTERNAL_TOPK if is_v1 else topk
    results = model.run(case.target_smiles, topk=mlp_topk)
    if not results:
        return None

    reactants_ = results["reactants"]
    scores = results["scores"]
    scores_ref = results.get("scores_reference", scores)

    ranked = sorted(
        zip(reactants_, scores, scores_ref),
        key=lambda x: x[1],
        reverse=True,
    )[:topk]

    gt_key = case.gt_reactants_key
    hit_info = (
        find_gt_hit(reactants_, gt_key, topk)
        if gt_key is not None
        else {"hit": False, "hit_rank": None, "hit_reactants": None}
    )

    reactions: list[ReactionSummary] = []
    total_prec = 0
    avail_prec = 0
    all_avail_reactions = 0

    for rank, (rxn, sc, sc_ref) in enumerate(ranked):
        precursors = mark_availability(rxn, starting_mols)
        n_avail = sum(1 for p in precursors if p["status"] == "available")
        n_unavail = len(precursors) - n_avail
        total_prec += len(precursors)
        avail_prec += n_avail
        all_avail = n_unavail == 0 and len(precursors) > 0
        if all_avail:
            all_avail_reactions += 1

        matches_gt = (
            prediction_matches_gt(rxn, gt_key) if gt_key is not None else False
        )

        reactions.append(
            ReactionSummary(
                rank=rank,
                score=float(sc),
                score_reference=float(sc_ref),
                reactants=rxn,
                matches_ground_truth=matches_gt,
                precursors=precursors,
                n_available=n_avail,
                n_unavailable=n_unavail,
                all_available=all_avail,
                any_available=n_avail > 0,
            )
        )

    frac_prec = (avail_prec / total_prec) if total_prec else 0.0
    frac_rxn_all = (
        all_avail_reactions / len(reactions) if reactions else 0.0
    )

    return MoleculeRun(
        route_index=case.route_index,
        target_smiles=case.target_smiles,
        checkpoint=checkpoint_key,
        topk=topk,
        n_predictions=len(reactions),
        ground_truth_reactants=case.gt_reactants,
        hit=hit_info["hit"],
        hit_rank=hit_info["hit_rank"],
        reactions=reactions,
        frac_precursors_available=frac_prec,
        frac_reactions_all_available=frac_rxn_all,
    )


def print_molecule_report(run: MoleculeRun) -> None:
    print(f"\n{'=' * 72}")
    print(
        f"route[{run.route_index}]  checkpoint={run.checkpoint}  topk={run.topk}"
    )
    print(
        f"target: {run.target_smiles[:100]}"
        f"{'...' if len(run.target_smiles) > 100 else ''}"
    )
    if run.ground_truth_reactants:
        print(f"ground truth: {run.ground_truth_reactants[:100]}...")
        hit_str = (
            f"HIT at rank {run.hit_rank}"
            if run.hit
            else "MISS (GT not in top-k)"
        )
        print(f"  {hit_str}")
    print(
        f"precursors available: {run.frac_precursors_available:.1%}  |  "
        f"reactions all-available: {run.frac_reactions_all_available:.1%}"
    )
    for rx in run.reactions:
        gt_mark = " [GT]" if rx.matches_ground_truth else ""
        prec_str = ", ".join(
            f"{p['smiles'][:36]}{'…' if len(p['smiles']) > 36 else ''} [{p['status']}]"
            for p in rx.precursors
        )
        print(
            f"  #{rx.rank}  score={rx.score:.4f}{gt_mark}  |  {prec_str}"
        )


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def aggregate_hit_rate(rows: list[dict], ck: str, topk: int) -> float | None:
    subset = [
        r
        for r in rows
        if r["checkpoint"] == ck
        and r["topk"] == topk
        and r.get("hit") is not None
        and r["n_predictions"] > 0
    ]
    if not subset:
        return None
    return sum(1 for r in subset if r["hit"]) / len(subset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiments: MLP one-step experiment with ground-truth hit@k"
    )
    parser.add_argument(
        "--checkpoints",
        default="v1,v2",
        help="Comma-separated: v1, v2, v3",
    )
    parser.add_argument(
        "--topk",
        default="5,10,20",
        help="Top-k values to compare (agent tool.topk)",
    )
    parser.add_argument(
        "-n",
        "--num-molecules",
        type=int,
        default=8,
        help="Number of routes from the start of the test pkl",
    )
    parser.add_argument(
        "--indices",
        default=None,
        help="Route indices in test pkl, e.g. 0,1,5,12",
    )
    parser.add_argument(
        "--smiles",
        nargs="*",
        default=None,
        help="Explicit SMILES list (skips ground-truth metrics)",
    )
    parser.add_argument("--test-pkl", type=Path, default=DEFAULT_TEST_PKL)
    parser.add_argument("--starting-mols", type=Path, default=None)
    parser.add_argument(
        "--cache-starting-mols",
        type=Path,
        default=REPO_ROOT / "dataset" / ".starting_mols_set.pkl",
    )
    parser.add_argument("--no-cache-starting-mols", action="store_true")
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="-1 = CPU; >=0 = CUDA device id",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    _add_repo_to_path()

    starting_path = args.starting_mols
    if starting_path is None:
        canonical = REPO_ROOT / "dataset" / "origin_dict_canonical.csv"
        origin = REPO_ROOT / "dataset" / "origin_dict.csv"
        starting_path = canonical if canonical.is_file() else origin

    cache_path = None if args.no_cache_starting_mols else args.cache_starting_mols

    print("Loading starting molecules…", flush=True)
    t0 = time.time()
    starting_mols = load_starting_mols(starting_path, cache_path)
    print(f"  {len(starting_mols):,} molecules in {time.time() - t0:.1f}s")

    if not args.test_pkl.is_file() and not args.smiles:
        raise FileNotFoundError(
            f"Test pkl not found: {args.test_pkl}\n"
            "Download retro_data.zip (see README.md)."
        )

    cases = load_test_cases(
        args.test_pkl, parse_int_list(args.indices) if args.indices else None,
        args.num_molecules, args.smiles,
    )
    has_gt = cases[0].gt_reactants_key is not None if cases else False
    print(f"Test molecules: {len(cases)}  (ground-truth: {'yes' if has_gt else 'no'})")

    ckpt_keys = parse_str_list(args.checkpoints)
    topk_values = parse_int_list(args.topk)

    from mlp_retrosyn.mlp_inference import MLPModel

    if not DEFAULT_TEMPLATE.is_file():
        raise FileNotFoundError(f"Templates not found: {DEFAULT_TEMPLATE}")

    all_runs: list[dict] = []
    summary_rows: list[dict] = []

    for ck in ckpt_keys:
        if ck not in CHECKPOINTS:
            raise KeyError(f"Unknown checkpoint {ck!r}; choose from {list(CHECKPOINTS)}")
        ckpt_path = CHECKPOINTS[ck]
        if not ckpt_path.is_file():
            print(f"SKIP {ck}: missing {ckpt_path}", file=sys.stderr)
            continue

        print(f"\nLoading MLP [{ck}] from {ckpt_path.name}…", flush=True)
        t_load = time.time()
        model = MLPModel(
            str(ckpt_path),
            str(DEFAULT_TEMPLATE),
            device=args.device,
        )
        print(f"  loaded in {time.time() - t_load:.1f}s")

        for topk in topk_values:
            for case in cases:
                t_run = time.time()
                run = run_one_molecule(model, case, ck, topk, starting_mols)
                elapsed = time.time() - t_run

                if run is None:
                    print(
                        f"  [{ck} topk={topk} idx={case.route_index}] "
                        f"no predictions ({elapsed:.2f}s)"
                    )
                    summary_rows.append(
                        {
                            "route_index": case.route_index,
                            "checkpoint": ck,
                            "topk": topk,
                            "n_predictions": 0,
                            "hit": None,
                            "hit_rank": None,
                            "frac_precursors_available": None,
                            "frac_reactions_all_available": None,
                            "inference_s": elapsed,
                        }
                    )
                    continue

                if not args.quiet:
                    print_molecule_report(run)

                row = {
                    "route_index": run.route_index,
                    "target_smiles": run.target_smiles,
                    "ground_truth_reactants": run.ground_truth_reactants,
                    "checkpoint": ck,
                    "topk": topk,
                    "n_predictions": run.n_predictions,
                    "hit": run.hit if has_gt else None,
                    "hit_rank": run.hit_rank,
                    "frac_precursors_available": run.frac_precursors_available,
                    "frac_reactions_all_available": run.frac_reactions_all_available,
                    "inference_s": round(elapsed, 3),
                }
                summary_rows.append(row)
                all_runs.append(
                    {
                        **row,
                        "reactions": [asdict(r) for r in run.reactions],
                    }
                )
                if args.quiet:
                    hit_s = (
                        f"hit@{topk} rank={run.hit_rank}"
                        if run.hit
                        else f"miss@{topk}"
                        if has_gt
                        else ""
                    )
                    print(
                        f"  [{ck} k={topk} idx={case.route_index}] "
                        f"{run.n_predictions} rxns, avail={run.frac_precursors_available:.0%}, "
                        f"{hit_s} {elapsed:.2f}s"
                    )

    print("\n" + "=" * 72)
    print("SUMMARY (mean over molecules per checkpoint × topk)")
    print("-" * 72)
    for ck in ckpt_keys:
        for topk in topk_values:
            subset = [
                r
                for r in summary_rows
                if r["checkpoint"] == ck
                and r["topk"] == topk
                and r["n_predictions"] > 0
            ]
            if not subset:
                continue
            mean_avail = sum(r["frac_precursors_available"] for r in subset) / len(
                subset
            )
            mean_all = sum(
                r["frac_reactions_all_available"] for r in subset
            ) / len(subset)
            mean_t = sum(r["inference_s"] for r in subset) / len(subset)
            hit_rate = aggregate_hit_rate(summary_rows, ck, topk)
            hit_s = f"  hit@{topk}={hit_rate:.1%}" if hit_rate is not None else ""
            print(
                f"  {ck:3s}  topk={topk:2d}  n={len(subset):2d}  "
                f"precursors_avail={mean_avail:.1%}  "
                f"rxns_all_avail={mean_all:.1%}  "
                f"mean_inference={mean_t:.2f}s{hit_s}"
            )

    if args.output:
        payload = {
            "config": {
                "checkpoints": ckpt_keys,
                "topk": topk_values,
                "starting_mols": str(starting_path),
                "test_pkl": str(args.test_pkl),
                "device": args.device,
                "ground_truth_matching": "canonical isomeric SMILES, sorted fragments",
            },
            "summary": summary_rows,
            "aggregate_hit_rate": {
                f"{ck}_topk_{k}": aggregate_hit_rate(summary_rows, ck, k)
                for ck in ckpt_keys
                for k in topk_values
            },
            "runs": all_runs,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
