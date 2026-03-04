# Reproducing HiVT, TNT, and DenseTNT on V2X-Seq

This documents how I trained and evaluated three trajectory prediction baselines on the V2X-Seq cooperative vehicle-infrastructure dataset. I ran everything on yzs1 using a single RTX A6000 GPU.

All three models originally target the Argoverse dataset. Getting them to work on V2X-Seq required several non-trivial fixes — mostly in evaluation code, not training. I've documented every issue I hit and how I fixed it.

---

## Setup

**Server:** yzs1  
**Conda envs:** HiVT (for HiVT), TNT (for TNT and DenseTNT)  
**GPU:** RTX A6000 (49GB) — single GPU for all runs

Key paths:
- Source data: `/scratch/muminul951/hivt_data`
- Preprocessed HiVT data: `/scratch/muminul951/hivt_data/cooperative-vehicle-infrastructure/fusion_for_prediction`
- Preprocessed TNT/DenseTNT data: `.../fusion_for_prediction/interm_data`
- V2X-Seq repo: `/scratch/muminul951/v2x/V2X-Graph/required/DAIR-V2X-Seq`
- DenseTNT repo: `/scratch/muminul951/v2x/DenseTNT`

---

## Results

| Model | minADE | minFDE | MR | Notes |
|---|---|---|---|---|
| HiVT | TBD | TBD | TBD | training in progress |
| TNT | 5.42m | 10.10m | 0.342 | |
| DenseTNT | 2.00m | 2.93m | 0.318 | Stage 1 + Stage 2 |

All evaluated on the val split (10,229 scenes), horizon=50, top_k=6. Each model predicts the target agent (agent index 0) per scene.

These numbers are not directly comparable to V2X-Graph paper results — they used a different dataset (V2X-Traj) and 8× RTX 3090s. These are standalone baselines on V2X-Seq.

---

## Preprocessing

There are two separate preprocessing pipelines and both need to run before training anything.

### For HiVT

```bash
cd /scratch/muminul951/v2x/V2X-Graph/required/DAIR-V2X-Seq

nohup python tools/trajectory_fusion/fusion_for_prediction.py \
    --data_root /scratch/muminul951/hivt_data --split train \
    > /scratch/muminul951/hivt_data/preprocess_train.log 2>&1 &

nohup python tools/trajectory_fusion/fusion_for_prediction.py \
    --data_root /scratch/muminul951/hivt_data --split val \
    > /scratch/muminul951/hivt_data/preprocess_val.log 2>&1 &
```

The script hardcodes the output to `data_root/cooperative-vehicle-infrastructure/tfd_fusion/`. I pointed `data_root` to my own scratch space and symlinked Maiqi's source data into it so the output goes to my directory, not theirs.

### For TNT and DenseTNT

Before running, create three map symlinks in the fusion_for_prediction folder — the preprocessing script expects them there:

```bash
export FP="/scratch/muminul951/hivt_data/cooperative-vehicle-infrastructure/fusion_for_prediction"
ln -s ${FP}/map_files/yizhuang_PEK_vector_map.json ${FP}/yizhuang_PEK_vector_map.json
ln -s ${FP}/map_files/yizhuang_PEK_halluc_bbox_table.npy ${FP}/yizhuang_PEK_halluc_bbox_table.npy
ln -s ${FP}/map_files/yizhuang_PEK_tableidx_to_laneid_map.json ${FP}/yizhuang_PEK_tableidx_to_laneid_map.json
```

The preprocessing script has `num_workers=4` by default which causes a deadlock. Open `core/util/preprocessor/tfd_preprocess.py` and change it to `num_workers=0`, then run:

```bash
cd /scratch/muminul951/v2x/V2X-Graph/required/DAIR-V2X-Seq/projects/TNT_plugin
python core/util/preprocessor/tfd_preprocess.py --root ${FP}
```

This produces `interm_data/train_intermediate/` (25,574 samples) and `interm_data/val_intermediate/` (10,229 samples).

**Important:** After preprocessing finishes, delete the `train_intermediate/processed/` folder and let it regenerate. The initial cache gets built from a partial run and only has ~7,434 samples instead of 25,574. PyTorch Geometric reads from it silently with no warning. I only caught this because TNT was running 117 batches per epoch instead of 400. To regenerate:

```bash
rm -rf interm_data/train_intermediate/processed/
```

Then just start training — it will rebuild the cache automatically on first run (takes about 10 minutes).

---

## HiVT

```bash
conda activate HiVT
cd /scratch/muminul951/v2x/V2X-Graph/required/DAIR-V2X-Seq/projects/HiVT_plugin

export DATA_ROOT="/scratch/muminul951/hivt_data/cooperative-vehicle-infrastructure/fusion_for_prediction"
```

### Training

```bash
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python -W ignore train.py \
    --root ${DATA_ROOT} \
    --embed_dim 128 \
    --check_val_every_n_epoch 1 \
    --num_workers 8 \
    --max_epochs 64 \
    --train_batch_size 16 \
    --val_batch_size 16 \
    --monitor val_minFDE \
    --save_top_k 5 \
    > /scratch/muminul951/hivt_data/hivt_train_128.log 2>&1 &
```

### Evaluation

Find the best checkpoint under `lightning_logs/version_20/checkpoints/` and run:

```bash
python eval.py \
    --root ${DATA_ROOT} \
    --batch_size 32 \
    --num_workers 4 \
    --ckpt_path lightning_logs/version_20/checkpoints/<best>.ckpt \
    2>&1 | tee /scratch/muminul951/hivt_data/hivt_eval.log
```

---

## TNT

```bash
conda activate TNT
cd /scratch/muminul951/v2x/V2X-Graph/required/DAIR-V2X-Seq/projects/TNT_plugin

export TNT_DATA_ROOT="/scratch/muminul951/hivt_data/cooperative-vehicle-infrastructure/fusion_for_prediction/interm_data"
```

### Code fix — eval displacement mismatch

This is the most important fix. Without it, eval reports minADE=9.95m which looks like a real result but is completely wrong.

The model predicts per-step displacements (each step is ~1m of movement). The eval code was applying `cumsum` to ground truth to convert it to absolute positions, but never applying `cumsum` to predictions. So it was comparing absolute positions (~56m final) against raw displacements (~1m per step). The 9.95m ADE was just the average of this mismatch, not actual model error.

The fix is in `core/trainer/tnt_trainer.py` around line 264. Change:

```python
# wrong
gt = data.y.unsqueeze(1).view(batch_size, -1, 2).cumsum(axis=1).numpy()

# correct — model was trained on raw displacements, compare in same space
gt = data.y.unsqueeze(1).view(batch_size, -1, 2).numpy()
```

How I confirmed the bug: printed raw predictions and gt for the first sample. Predictions were ~0.1m per step (correct for displacements). GT after cumsum was 56m final position. GT without cumsum was ~1.2m final step displacement, and minFDE vs raw gt was 1.13m instead of 56.78m.

### Training

```bash
PYTHONWARNINGS="ignore" nohup python -W ignore train.py \
    --data_root ${TNT_DATA_ROOT} \
    --output_dir /scratch/muminul951/hivt_data/tnt_checkpoints_retrain \
    --batch_size 64 \
    --n_epoch 64 \
    --num_workers 0 \
    --horizon 50 \
    --top_k 6 \
    --local_rank 0 \
    > /scratch/muminul951/hivt_data/tnt_retrain.log 2>&1 &
```

Use `--num_workers 0`. Any higher value causes a deadlock when loading the 46GB processed cache.

### Evaluation

```bash
nohup python eval.py \
    -r ${TNT_DATA_ROOT} \
    -s val -b 64 -w 0 \
    --horizon 50 --top_k 6 \
    -c -cd 0 \
    -rm /scratch/muminul951/hivt_data/tnt_checkpoints_retrain/03-04-03-25/final_TNT.pth \
    -d /scratch/muminul951/hivt_data/tnt_eval_results \
    > /scratch/muminul951/hivt_data/tnt_eval.log 2>&1 &
```

Results: minADE=5.42m, minFDE=10.10m, MR=0.342

TNT's numbers are noticeably worse than DenseTNT. I verified this is not a training or data bug — I confirmed full 25,574 training samples were used (400 batches/epoch), loss converged normally, and eval is comparing the right things after the cumsum fix. TNT just struggles on V2X-Seq. The goal scoring head never meaningfully converged across multiple training runs, which DenseTNT's more sophisticated architecture handles much better.

---

## DenseTNT

DenseTNT uses a two-stage training process. Stage 1 trains everything except the goal set predictor for 16 epochs. Stage 2 trains only the goal set predictor for 6 epochs, starting from the Stage 1 checkpoint.

```bash
conda activate TNT
cd /scratch/muminul951/v2x/DenseTNT

export DATA_ROOT="/scratch/muminul951/hivt_data/cooperative-vehicle-infrastructure/fusion_for_prediction/interm_data"
```

### One-time setup

Compile the Cython extension first — training will crash at import without it:

```bash
cd src/
pip install cython
cython -a utils_cython.pyx && python setup.py build_ext --inplace
cd ..
```

The codebase assumes Argoverse throughout. Three files need changes to work with V2X-Seq data.

**In `src/run.py`:** Change both dataset imports (around lines 246 and 294) from `dataset_argoverse` to `dataset_v2xseq`. Also remove the line `assert args.train_batch_size == 64` — it blocks training with any other batch size.

**In `src/do_eval.py`:** Change both imports (around lines 59 and 136) from `dataset_argoverse` to `dataset_v2xseq`.

**In `src/dataset_v2xseq.py`** there are four issues to fix:

1. The `cent_x` and `cent_y` fields in the mapping dict were set to the UTM origin coordinates (`orig[0]`, `orig[1]` ≈ 417894, 4730251). DenseTNT's `to_origin_coordinate()` function uses these to transform predictions into a global frame, which produced minADE=4,749,687m. V2X-Seq data is already in local agent-centric frame, so set `cent_x=0.0`, `cent_y=0.0`, `angle=0.0`.

2. The `origin_labels` key was missing from the mapping dict. The eval code accesses it directly. Add `origin_labels=labels.astype(np.float32)` alongside the existing `labels` entry.

3. The `file_name` field was set to just `seq_id` (e.g. `"10"`). The eval code does `int(filename[:-4])` expecting something like `"10.csv"`. Stripping 4 characters from `"10"` gives `""` which can't be converted to int. Set `file_name=seq_id + '.csv'`.

4. The `post_eval` function doesn't exist in `dataset_v2xseq.py` but `do_eval.py` imports and calls it. The original in `dataset_argoverse.py` calls the Argoverse evaluation API which doesn't apply here. I added a custom `post_eval` to `dataset_v2xseq.py` that computes minADE, minFDE, and MR directly from predictions and ground truth.

### Stage 1 Training

Must run from the repo root, not from inside `src/` — the script does `os.chdir('src/')` internally:

```bash
echo y | CUDA_VISIBLE_DEVICES=3 nohup python src/run.py \
    --argoverse --future_frame_num 50 --do_train -d 1 \
    --data_dir ${DATA_ROOT} \
    --output_dir /scratch/muminul951/hivt_data/densetnt_output \
    --hidden_size 128 --train_batch_size 16 --use_map --core_num 4 \
    --other_params semantic_lane direction l1_loss goals_2D \
        enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
        lane_scoring complete_traj complete_traj-3 \
    > /scratch/muminul951/hivt_data/densetnt_train.log 2>&1 &
```

The `echo y |` is needed because the script prompts to confirm overwriting the output directory, which blocks under nohup. The `-d 1` flag limits to single GPU — the default `-d 8` tries to spawn 8 DDP processes which OOMs even on a 49GB card.

Verify the Stage 1 checkpoint exists before moving to Stage 2:
```bash
ls /scratch/muminul951/hivt_data/densetnt_output/model_save/model.16.bin
```

### Stage 2 Training

```bash
export MODEL_PATH="/scratch/muminul951/hivt_data/densetnt_output/model_save/model.16.bin"

echo y | CUDA_VISIBLE_DEVICES=3 nohup python src/run.py \
    --argoverse --future_frame_num 50 --do_train -d 1 \
    --num_train_epochs 6 \
    --data_dir ${DATA_ROOT} \
    --output_dir /scratch/muminul951/hivt_data/densetnt_output_stage2 \
    --hidden_size 128 --train_batch_size 16 --use_map --core_num 4 \
    --other_params semantic_lane direction l1_loss goals_2D \
        enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
        lane_scoring complete_traj \
        set_predict=6 set_predict-6 data_ratio_per_epoch=0.4 \
        set_predict-topk=0 set_predict-one_encoder set_predict-MRratio=0.0 \
        set_predict-train_recover=${MODEL_PATH} \
    > /scratch/muminul951/hivt_data/densetnt_stage2.log 2>&1 &
```

Note `complete_traj-3` is removed for Stage 2 — this follows the original DenseTNT paper's setup.

### Evaluation

```bash
echo y | CUDA_VISIBLE_DEVICES=3 nohup python src/run.py \
    --argoverse --future_frame_num 50 --do_eval -d 1 \
    --data_dir ${DATA_ROOT} --data_dir_for_val ${DATA_ROOT} \
    --output_dir /scratch/muminul951/hivt_data/densetnt_output_stage2 \
    --hidden_size 128 --eval_batch_size 16 --use_map --core_num 4 \
    --model_recover_path /scratch/muminul951/hivt_data/densetnt_output_stage2/model_save/model.6.bin \
    --other_params semantic_lane direction l1_loss goals_2D \
        enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
        lane_scoring complete_traj \
        set_predict=6 set_predict-6 data_ratio_per_epoch=0.4 \
        set_predict-topk=0 set_predict-one_encoder set_predict-MRratio=0.0 \
        set_predict-train_recover=/scratch/muminul951/hivt_data/densetnt_output/model_save/model.16.bin \
    > /scratch/muminul951/hivt_data/densetnt_eval.log 2>&1 &
```

Results: minADE=2.00m, minFDE=2.93m, MR=0.318
