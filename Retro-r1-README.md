# Retro-R1 for Retrosynthetic Planning

This repository is based on the open-source codebase of [Retro*](https://github.com/binghong-ml/retro_star) and [Agent-R1](https://github.com/0russwest0/Agent-R1).

For **local setup, data download, rdchiral tests, and Experiments (one-step MLP on CPU)**, see [README.md](README.md).

## Setup

### Install dependencies (GPU training)

Start with the base environment from [README.md](README.md) (`uv sync`). Then add training-specific packages:

```bash
cd Retro-R1
uv sync
pip install flash-attn==2.7.4.post1
unzip -q verl.zip
cd verl
pip install --no-deps -e .
cd ..
```

`verl.zip` is included in the repository. `mlp_retrosyn` and `rdchiral` are already installed by `uv sync` (editable packages under `packages/`).

> **Note:** There is no `environment.yml` in this repo. If you prefer conda, create a Python 3.10 env manually and install the same packages.

### Download the necessary files

To reproduce the results in the paper, download the training dataset, evaluation datasets (USPTO, ChEMBL-1000), starting molecules, and template rules. See [README.md — Step 2](README.md#step-2--download-required-files) for the full bundle ([RETRO-R1-DATA.zip](https://drive.google.com/file/d/1ESkk0spmM1C7Z-b38mGF7cEuH-5l77QP/view?usp=sharing)) or the minimal Retro* bundle.

Additional details:

- Retro* V1 weights: [retro_data.zip](https://www.dropbox.com/s/ar9cupb18hv96gj/retro_data.zip?dl=0) → extract `dataset/` and `one_step_model/` into the repo root.
- V2 / V3 one-step weights: [Google Drive folder](https://drive.google.com/drive/u/0/folders/13DdftEV0x55OZ8ZxHNAkmcvi_4x90hPI) → `one_step_model/retro_star_value_ours.ckpt` (V2), `one_step_model/retro_star_zero_ours.ckpt` (V3).
- V4 is not released; train following [PDVN](https://github.com/DiXue98/PDVN) or request weights. Save as `one_step_model/retro_star_V4.ckpt`.
- ChEMBL-1000 testset: [Google Drive](https://drive.google.com/drive/folders/198WuPlSyMeMvvd4i2SM833jPAcGllzDu?usp=sharing) → `dataset/`.

## Preprocess the dataset

```bash
python ./examples/data_preprocess/reaction.py
```

## Training

```bash
bash training_script.sh
```

This script uses one node with eight GPUs. For multi-node setup, follow the [verl](https://github.com/volcengine/verl) instructions.

## Export model

Edit `export.sh` — set `ori_pth`, `ckpt_pth`, and `export_pth` to your checkpoint paths — then run:

```bash
bash export.sh
```

## Evaluation on Retro*-190 testset

```bash
bash test_retro_script.sh
```

## Evaluation on ChEMBL-1000 testset

```bash
bash test_chembl_script.sh
```
