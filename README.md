# Retro-R1 (My minimal experiments)

Retrosynthetic planning based on [Retro*](https://github.com/binghong-ml/retro_star) and [Agent-R1](https://github.com/0russwest0/Agent-R1).

For full paper reproduction (training, LLM agent evaluation), see [Retro-r1-README.md](Retro-r1-README.md).

This document covers **local setup**, **downloading data**, and **Experiments** — a CPU-friendly one-step MLP benchmark without an LLM. You can run it via CLI scripts or the interactive notebook [`experiments_mlp_one_step.ipynb`](experiments_mlp_one_step.ipynb).

---

## Prerequisites

- **Python** ≥ 3.10  
- **[uv](https://docs.astral.sh/uv/)** (recommended) or conda  
- **~15 GB disk** for `dataset/` and `one_step_model/` after download  
- **RAM**: ~4 GB for the MLP + ~2 GB for the starting-molecules cache (first run builds `dataset/.starting_mols_set.pkl`)

On Apple Silicon, inference uses **CPU** (`device=-1`).

---

## Step 1 — Clone and install Python dependencies

```bash
git clone https://github.com/arqoofficial/Retro-R1.git Retro-R1
cd Retro-R1
uv sync
```

See [Retro-r1-README.md](Retro-r1-README.md) for `verl`, `flash-attn`, and multi-GPU training.

---

## Step 2 — Download required files

Large assets are **not** in git (see `.gitignore`). Download them into the repo root.

### 2.1 Full data bundle (recommended)

| Source | [RETRO-R1-DATA.zip](https://drive.google.com/file/d/1ESkk0spmM1C7Z-b38mGF7cEuH-5l77QP/view?usp=sharing) (Google Drive) |
|--------|--------------------------------------------------------------------------------------------------------------------------|
| Size | ~15 GB compressed, ~15 GB after extract |
| Contents | Everything from `dataset/` and `one_step_model/` — V1/V2/V3 checkpoints, routes, `origin_dict.csv`, pre-built `.starting_mols_set.pkl`, training assets |

**Download:** open the link in a browser and save `RETRO-R1-DATA.zip` into the repo root (`Retro-R1/`).

**Extract (fastest):** run from the repo root — the archive already contains top-level `dataset/` and `one_step_model/` folders:

```bash
cd Retro-R1
unzip -q RETRO-R1-DATA.zip
```

`-q` skips per-file progress spam; extraction takes a few minutes on SSD. You can delete the zip after [verify](#24-verify-layout).

**macOS (Finder):** double-click the zip — it unpacks next to the archive. Move `dataset/` and `one_step_model/` into `Retro-R1/` if they landed elsewhere.

**Windows (PowerShell):**

```powershell
cd Retro-R1
Expand-Archive -Path RETRO-R1-DATA.zip -DestinationPath .
```

**Optional CLI download** (if browser download is slow or interrupted):

```bash
uv pip install gdown
gdown "https://drive.google.com/uc?id=1ESkk0spmM1C7Z-b38mGF7cEuH-5l77QP" -O RETRO-R1-DATA.zip
unzip -q RETRO-R1-DATA.zip
```

### 2.2 Retro* bundle (minimal alternative)

If you only need the Experiments smoke test (V1), use the smaller Retro* zip instead:

| Source | [retro_data.zip](https://www.dropbox.com/s/ar9cupb18hv96gj/retro_data.zip?dl=0) |
|--------|-----------------------------------------------------------------------------------|
| Extract | Same as above: `unzip -q retro_data.zip` at repo root → `dataset/` + `one_step_model/` |

**Minimum files for Experiments:**

| Path | Purpose |
|------|---------|
| `one_step_model/saved_rollout_state_1_2048.ckpt` | V1 one-step policy |
| `one_step_model/template_rules_1.dat` | Reaction templates |
| `dataset/routes_possible_test_hard.pkl` | Retro*-190 test routes (ground truth) |
| `dataset/origin_dict.csv` | Starting molecules (~23M) for available/unavailable labels |

Optional: `dataset/origin_dict_canonical.csv` if provided in your zip (script falls back to `origin_dict.csv`).

### 2.3 Extended one-step checkpoints (optional)

Included in [RETRO-R1-DATA.zip](#21-full-data-bundle-recommended). Download separately only if you used the minimal Dropbox bundle:

| File | Version | Source |
|------|---------|--------|
| `one_step_model/retro_star_value_ours.ckpt` | V2 | [Google Drive folder](https://drive.google.com/drive/u/0/folders/13DdftEV0x55OZ8ZxHNAkmcvi_4x90hPI) |
| `one_step_model/retro_star_zero_ours.ckpt` | V3 | same folder |

### 2.4 Verify layout

```bash
ls one_step_model/saved_rollout_state_1_2048.ckpt \
   one_step_model/template_rules_1.dat \
   dataset/routes_possible_test_hard.pkl \
   dataset/origin_dict.csv
```

Expected tree (abbreviated):

```
Retro-R1/
├── dataset/
│   ├── origin_dict.csv
│   └── routes_possible_test_hard.pkl
├── one_step_model/
│   ├── saved_rollout_state_1_2048.ckpt
│   ├── template_rules_1.dat
│   ├── retro_star_value_ours.ckpt   # optional V2
│   └── retro_star_zero_ours.ckpt    # optional V3
├── experiments_mlp_one_step.ipynb   # interactive experiment + plots
├── packages/
│   ├── mlp_retrosyn/
│   └── rdchiral/
└── scripts/
    ├── experiments_mlp_one_step.py
    └── visualize_experiments_mlp.py
```

---

## Step 3 — Run unit tests (rdchiral)

`rdchiral` applies templates during MLP inference. Run its test suite after install:

```bash
cd Retro-R1
uv run pytest packages/rdchiral/test/test_rdchiral.py -v
```

Quick smoke test (first case only):

```bash
uv run pytest packages/rdchiral/test/test_rdchiral.py -v -k case_0
```

---

## Step 4 — Experiments: one-step MLP experiment

Script: [`scripts/experiments_mlp_one_step.py`](scripts/experiments_mlp_one_step.py)

Run all commands below from the **repository root** (`Retro-R1/`). The MLP loads a V1 reference checkpoint via the relative path `./one_step_model/...`.

**What it measures**

1. **top-k** reactions (5 / 10 / 20) — same values as in agent training (`tool.topk` defaults to 5; experiments sweep 5, 10, and 20).  
2. **Checkpoints** — `v1` (Retro* default), `v2` (`retro_star_value_ours.ckpt`), `v3` (`retro_star_zero_ours.ckpt`; optional, Google Drive).  
3. **Stock labels** — each precursor vs `origin_dict.csv` → available / unavailable.  
4. **Ground truth** — first step of each route in `routes_possible_test_hard.pkl` (`product>>reactants`).  
   - **hit@k**: canonical isomeric SMILES match (sorted fragment multiset) in top-k predictions.

**First run** loads ~23M starting molecules (~20 s); later runs use `dataset/.starting_mols_set.pkl`.

### Quick smoke test (2 molecules, V1 only)

```bash
uv run python scripts/experiments_mlp_one_step.py \
  --checkpoints v1 \
  --topk 5 \
  -n 2 \
  --quiet
```

### Full comparison (V1 vs V2, top-k 5/10/20, 10 molecules)

```bash
mkdir -p results
uv run python scripts/experiments_mlp_one_step.py \
  --checkpoints v1,v2 \
  --topk 5,10,20 \
  -n 10 \
  --output results/experiments_mlp.json
```

### Full comparison (V1 vs V2 vs V3)

Requires V2 and V3 from [RETRO-R1-DATA.zip](#21-full-data-bundle-recommended) or [Google Drive](#23-extended-one-step-checkpoints-optional). Each checkpoint loads separately — expect roughly 3× the runtime of V1-only.

```bash
mkdir -p results
uv run python scripts/experiments_mlp_one_step.py \
  --checkpoints v1,v2,v3 \
  --topk 5,10,20 \
  -n 10 \
  --output results/experiments_mlp.json

uv run python scripts/visualize_experiments_mlp.py
```

Plots show three bars/series per top-k (V1 / V2 / V3). Missing checkpoints are skipped with `SKIP vN: missing ...` on stderr.

### Specific route indices

```bash
uv run python scripts/experiments_mlp_one_step.py \
  --indices 0,1,5,12,42 \
  --checkpoints v1,v2 \
  --topk 5,10,20
```

### Custom SMILES (no ground-truth metrics)

```bash
uv run python scripts/experiments_mlp_one_step.py \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --checkpoints v1 \
  --topk 5
```

### Example summary line

```
  v1   topk= 5  n=10  precursors_avail=22.6%  rxns_all_avail=0.0%  mean_inference=0.36s  hit@5=30.0%
```

### Notes

- **V1 vs tool behavior**: for checkpoint `v1`, the script calls `MLPModel.run(..., topk=50)` internally, then keeps the requested k — same as `SingleStepRetroTool`.  
- **Reference filter**: `MLPModel` always ranks with a reference net loaded from V1 (`realistic_filter`), even when the main checkpoint is V2.  
- **Results JSON** includes per-molecule `hit`, `hit_rank`, and `matches_ground_truth` per reaction.

### Interactive notebook (alternative to Steps 4–5)

Notebook: [`experiments_mlp_one_step.ipynb`](experiments_mlp_one_step.ipynb)

Same benchmark as the CLI scripts, in a Jupyter workflow: edit parameters in one cell, run MLP inference, inspect per-molecule tables, and render all five plots inline. Use it for interactive exploration; for batch runs or automation, prefer the scripts above.

**Launch** (from repo root; `jupyter` is installed via `uv sync`):

```bash
cd Retro-R1
uv run jupyter notebook experiments_mlp_one_step.ipynb
# or: uv run jupyter lab experiments_mlp_one_step.ipynb
```

If you open the notebook from a subdirectory, the first code cell still resolves `REPO_ROOT` automatically.

**Notebook structure** — run cells top to bottom on first use; later you can re-run only the sections you need:

| Section | Purpose |
|---------|---------|
| 1. Imports and paths | Checkpoints, test routes, default output path |
| 2. Plot functions | Same charts as `visualize_experiments_mlp.py` |
| 3. Experiment helpers | SMILES matching, stock lookup, MLP runner |
| 4. Configuration | Edit the `CONFIG` dict (see below) |
| 5. Run experiment | Inference + per-molecule summary table; writes JSON |
| 6. Aggregated metrics | Mean hit rate, stock fractions, inference time |
| Visualizations | Five separate cells (hit rate, heatmap, stock, rank hist, latency) |

**`CONFIG` keys** (section 4):

| Key | Default | Meaning |
|-----|---------|---------|
| `checkpoints` | `["v1", "v2"]` | MLP weights to compare (`v1`, `v2`, `v3`) |
| `topk` | `[5, 10, 20]` | Prediction list sizes |
| `num_molecules` | `8` | First N routes from `routes_possible_test_hard.pkl` |
| `indices` | `None` | Specific route indices, e.g. `[0, 1, 5, 12]` |
| `smiles` | `None` | Custom SMILES list (no ground-truth hit@k) |
| `skip_run` | `False` | `True` → load existing JSON, skip MLP inference |
| `heatmap_topk` | `10` | top-k for per-route heatmap and rank histogram |
| `output` | `results/experiments_mlp.json` | Result file (same format as the CLI script) |

**Plot-only mode:** after a script or notebook run produced JSON, set `skip_run=True` in section 4, re-run section 5, then run individual plot cells at the bottom.

Output is interchangeable: JSON from the notebook works with `visualize_experiments_mlp.py`, and JSON from the script can be loaded in the notebook with `skip_run=True`.

---

## Step 5 — Visualize experiment results

Script: [`scripts/visualize_experiments_mlp.py`](scripts/visualize_experiments_mlp.py)

Builds plots and a text summary from the JSON produced in [Step 4](#step-4--experiments-one-step-mlp-experiment).

**Defaults:** reads `results/experiments_mlp.json`, writes to `results/experiments_plots/`.

### After a full experiment run

```bash
uv run python scripts/visualize_experiments_mlp.py
```

Run this after Step 4 with `--output results/experiments_mlp.json` (or use the default path).

### Custom input / output

```bash
uv run python scripts/visualize_experiments_mlp.py \
  --input results/experiments_mlp.json \
  --output-dir results/experiments_plots \
  --heatmap-topk 10
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--input` | `results/experiments_mlp.json` | JSON from `experiments_mlp_one_step.py` |
| `--output-dir` | `results/experiments_plots/` | Directory for PNGs and CSV |
| `--heatmap-topk` | `10` | top-k for per-route heatmap and rank histogram |

### Output files

| File | Description |
|------|-------------|
| `aggregated_metrics.csv` | Mean hit rate, ranks, stock fractions, inference time per checkpoint × top-k |
| `01_hit_rate_by_topk.png` | Bar chart: hit@k by checkpoint |
| `02_per_route_top{k}.png` | Per-route hit/miss heatmap and hit-rank matrix |
| `03_precursor_availability.png` | Mean precursor availability in top-k |
| `04_hit_rank_dist_top{k}.png` | Histogram of hit ranks (hits only) |
| `05_inference_time.png` | Mean inference latency vs top-k |

The script also prints a text summary to stdout (same metrics as the experiment summary lines).

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `FileNotFoundError` for ckpt / pkl | Download and extract [RETRO-R1-DATA.zip](#21-full-data-bundle-recommended) ([Step 2](#step-2--download-required-files)). |
| `ModuleNotFoundError: rdkit` | Run via `uv run` or activate the env where `uv sync` installed deps. |
| Segfault on import | Run outside restricted sandboxes; needs native RDKit + PyTorch. |
| Very slow first start | Normal: building `.starting_mols_set.pkl` from `origin_dict.csv`. |
| `SKIP v2: missing ...` / `SKIP v3: missing ...` | Download V2/V3 from Google Drive, or drop missing keys from `--checkpoints` (e.g. `v1,v2` only). |
| `Results not found` in visualize script | Run Step 4 with `--output results/experiments_mlp.json` first, or pass `--input` to your JSON path. |

---

## Related documentation

- [Retro-r1-README.md](Retro-r1-README.md) — training, agent evaluation, ChEMBL-1000  
- [packages/rdchiral/README.md](packages/rdchiral/README.md) — template application details
