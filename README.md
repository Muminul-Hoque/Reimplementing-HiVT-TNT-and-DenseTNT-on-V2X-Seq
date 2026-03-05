# Reimplementing HiVT, TNT, and DenseTNT on V2X-Seq

This repo contains instructions and code to run HiVT, TNT, and DenseTNT as trajectory prediction baselines on the [V2X-Seq](https://github.com/AIR-THU/DAIR-V2X-Seq) dataset (cooperative vehicle-infrastructure subset, TFD split). All three models were originally designed for Argoverse. Getting them working on V2X-Seq required fixing several bugs in the evaluation pipelines — coordinate frame mismatches, missing dataset adapter functions, and environment compatibility issues.

## Results

Evaluated on the V2X-Seq val split (10,229 scenes), prediction horizon = 50 steps, K = 6 modes. Each model predicts the target agent only.

| Model | minADE (m) | minFDE (m) | MR |
|---|---|---|---|
| TNT | 5.42 | 10.10 | 0.342 |
| DenseTNT | 2.00 | 2.93 | 0.318 |
| HiVT | 1.10 | 1.91 | 0.286 |

> These numbers are not directly comparable to the V2X-Graph paper — they used a different split (V2X-Traj) on 8× RTX 3090s. These are standalone baselines on V2X-Seq TFD using a single RTX A6000.

## Code

Clone these forks — all fixes are already applied. The fixes sections below document what was changed and why.

| Model | Repository |
|---|---|
| HiVT + TNT | [Muminul-Hoque/DAIR-V2X-Seq](https://github.com/Muminul-Hoque/DAIR-V2X-Seq) |
| DenseTNT | [Muminul-Hoque/DenseTNT](https://github.com/Muminul-Hoque/DenseTNT) |

## Dataset

Download V2X-Seq from the [official repo](https://github.com/AIR-THU/DAIR-V2X-Seq). You need the cooperative-vehicle-infrastructure subset with the TFD (Trajectory Forecasting Dataset) split.

Expected structure after download:
```
data_root/
  cooperative-vehicle-infrastructure/
    cooperative-trajectories/
    infrastructure-trajectories/
    vehicle-trajectories/
    fusion_for_prediction/
      map_files/
```

## Environment Setup

```bash
# HiVT
conda env create -f envs/hivt.yaml
conda activate HiVT

# TNT and DenseTNT share the same env
conda env create -f envs/tnt.yaml
conda activate TNT
```

## Preprocessing

Two separate pipelines are required.

### Step 1 — Trajectory Fusion (for HiVT)

```bash
cd DAIR-V2X-Seq/

python tools/trajectory_fusion/fusion_for_prediction.py \
    --data_root /your/data_root --split train

python tools/trajectory_fusion/fusion_for_prediction.py \
    --data_root /your/data_root --split val
```

### Step 2 — TNT Preprocessing (for TNT and DenseTNT)

Create three map symlinks before running — the script expects them in the `fusion_for_prediction/` folder:

```bash
export FP="/your/data_root/cooperative-vehicle-infrastructure/fusion_for_prediction"
ln -s ${FP}/map_files/yizhuang_PEK_vector_map.json ${FP}/yizhuang_PEK_vector_map.json
ln -s ${FP}/map_files/yizhuang_PEK_halluc_bbox_table.npy ${FP}/yizhuang_PEK_halluc_bbox_table.npy
ln -s ${FP}/map_files/yizhuang_PEK_tableidx_to_laneid_map.json ${FP}/yizhuang_PEK_tableidx_to_laneid_map.json
```

Then run:

```bash
cd DAIR-V2X-Seq/projects/TNT_plugin/
python core/util/preprocessor/tfd_preprocess.py --root ${FP}
```

**After preprocessing**, delete the PyTorch Geometric cache and let it rebuild:

```bash
rm -rf interm_data/train_intermediate/processed/
```

The initial cache is built from a partial run and only contains ~7,434 of the 25,574 training samples. PyTorch Geometric reads it silently without any error. Training on it produces wrong results.

---

## HiVT

### Training

```bash
cd DAIR-V2X-Seq/projects/HiVT_plugin/

export DATA_ROOT="/your/data_root/cooperative-vehicle-infrastructure/fusion_for_prediction"

python train.py \
    --root ${DATA_ROOT} \
    --embed_dim 128 \
    --check_val_every_n_epoch 1 \
    --num_workers 8 \
    --max_epochs 64 \
    --train_batch_size 16 \
    --val_batch_size 16 \
    --monitor val_minFDE \
    --save_top_k 5
```

### Evaluation

Find the best checkpoint using tensorboard or by reading `lightning_logs/version_X/events.out.tfevents.*`, then:

```bash
python eval.py \
    --root ${DATA_ROOT} \
    --batch_size 32 \
    --num_workers 4 \
    --ckpt_path lightning_logs/version_X/checkpoints/best.ckpt
```

---

## TNT

### Training

```bash
cd DAIR-V2X-Seq/projects/TNT_plugin/

export TNT_DATA_ROOT="/your/data_root/cooperative-vehicle-infrastructure/fusion_for_prediction/interm_data"

python train.py \
    --data_root ${TNT_DATA_ROOT} \
    --output_dir /your/output_dir/tnt_checkpoints \
    --batch_size 64 \
    --n_epoch 64 \
    --num_workers 0 \
    --horizon 50 \
    --top_k 6 \
    --local_rank 0
```

`--num_workers 0` is required — higher values deadlock when loading the processed cache.

### Evaluation

```bash
python eval.py \
    -r ${TNT_DATA_ROOT} \
    -s val -b 64 -w 0 \
    --horizon 50 --top_k 6 \
    -c -cd 0 \
    -rm /your/output_dir/tnt_checkpoints/<timestamp>/final_TNT.pth \
    -d /your/output_dir/tnt_eval_results
```

---

## DenseTNT

### One-time Setup

```bash
cd DenseTNT/src/
pip install cython
cython -a utils_cython.pyx && python setup.py build_ext --inplace
cd ..
```

### Stage 1 Training (16 epochs)

Run from the repo root, not from inside `src/`:

```bash
cd DenseTNT/

export DATA_ROOT="/your/data_root/cooperative-vehicle-infrastructure/fusion_for_prediction/interm_data"

echo y | python src/run.py \
    --argoverse --future_frame_num 50 --do_train -d 1 \
    --data_dir ${DATA_ROOT} \
    --output_dir /your/output_dir/densetnt_stage1 \
    --hidden_size 128 --train_batch_size 16 --use_map --core_num 4 \
    --other_params semantic_lane direction l1_loss goals_2D \
        enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
        lane_scoring complete_traj complete_traj-3
```

The `-d 1` flag is required. The default `-d 8` spawns 8 DDP processes and OOMs even on a 49GB GPU.

### Stage 2 Training (6 epochs)

```bash
export MODEL_PATH="/your/output_dir/densetnt_stage1/model_save/model.16.bin"

echo y | python src/run.py \
    --argoverse --future_frame_num 50 --do_train -d 1 \
    --num_train_epochs 6 \
    --data_dir ${DATA_ROOT} \
    --output_dir /your/output_dir/densetnt_stage2 \
    --hidden_size 128 --train_batch_size 16 --use_map --core_num 4 \
    --other_params semantic_lane direction l1_loss goals_2D \
        enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
        lane_scoring complete_traj \
        set_predict=6 set_predict-6 data_ratio_per_epoch=0.4 \
        set_predict-topk=0 set_predict-one_encoder set_predict-MRratio=0.0 \
        set_predict-train_recover=${MODEL_PATH}
```

`complete_traj-3` is intentionally dropped for Stage 2, following the original DenseTNT paper.

### Evaluation

```bash
echo y | python src/run.py \
    --argoverse --future_frame_num 50 --do_eval -d 1 \
    --data_dir ${DATA_ROOT} --data_dir_for_val ${DATA_ROOT} \
    --output_dir /your/output_dir/densetnt_stage2 \
    --hidden_size 128 --eval_batch_size 16 --use_map --core_num 4 \
    --model_recover_path /your/output_dir/densetnt_stage2/model_save/model.6.bin \
    --other_params semantic_lane direction l1_loss goals_2D \
        enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
        lane_scoring complete_traj \
        set_predict=6 set_predict-6 data_ratio_per_epoch=0.4 \
        set_predict-topk=0 set_predict-one_encoder set_predict-MRratio=0.0 \
        set_predict-train_recover=/your/output_dir/densetnt_stage1/model_save/model.16.bin
```

---

## TNT Fixes

Two files were modified in `projects/TNT_plugin/`. All changes are already applied in [Muminul-Hoque/DAIR-V2X-Seq](https://github.com/Muminul-Hoque/DAIR-V2X-Seq).

**`core/trainer/tnt_trainer.py`**

1. **apex removed** — apex was unavailable on the training server. Replaced with standard PyTorch `DistributedDataParallel`.

2. **cumsum fix** — eval was applying cumsum to ground truth positions, comparing absolute positions against raw displacement predictions. Without this fix, eval reports minADE ≈ 9.95m which is not a real metric. Removed cumsum from gt to match the prediction coordinate frame, giving the correct 5.42m.

3. **num_workers=0** — the default value of 32 caused a deadlock when loading the processed cache.

**`core/util/preprocessor/tfd_preprocess.py`**

1. **num_workers=0** — same deadlock issue during preprocessing, fixed here as well.

---

## DenseTNT Fixes

Four bugs were fixed in `src/dataset_v2xseq.py`, plus import changes in `run.py` and `do_eval.py`. All changes are already applied in [Muminul-Hoque/DenseTNT](https://github.com/Muminul-Hoque/DenseTNT).

**`src/run.py` and `src/do_eval.py`**

Both import `dataset_argoverse` by default. Changed to `dataset_v2xseq` in all four import locations. Also removed the `assert args.train_batch_size == 64` line which blocked training with other batch sizes.

**`src/dataset_v2xseq.py`**

1. **Coordinate frame** — `cent_x` and `cent_y` were set to the UTM origin coordinates (~417894, 4730251). DenseTNT's `to_origin_coordinate()` uses these to transform predictions into global frame, producing minADE ≈ 4,749,687m. V2X-Seq data is already in local agent-centric frame, so `cent_x=0.0`, `cent_y=0.0`, `angle=0.0`.

2. **Missing `origin_labels`** — `do_eval.py` accesses `mapping[i]['origin_labels']` which was not set in the adapter. Added `origin_labels=labels.astype(np.float32)` to the mapping dict.

3. **`file_name` format** — `do_eval.py` parses the filename as `int(filename[:-4])`, expecting format `"10.csv"`. The adapter was setting `file_name=seq_id` (just `"10"`), so stripping 4 characters produces `""`. Fixed by setting `file_name=seq_id + '.csv'`.

4. **Missing `post_eval`** — `do_eval.py` imports `post_eval` from the dataset module. The original in `dataset_argoverse.py` calls the Argoverse1 evaluation API which is unavailable for V2X-Seq. A custom `post_eval` was added that computes minADE, minFDE, and MR directly.

---

## Checkpoints

Pretrained checkpoints are available at: *(add link)*

| Model | File | val_minFDE |
|---|---|---|
| HiVT (embed_dim=128) | hivt_epoch63.ckpt | 1.911 |
| TNT | final_TNT.pth | — |
| DenseTNT Stage 2 | model.6.bin | — |

---

## Citation

If you use this code, please cite the original papers:

```bibtex
@inproceedings{zhou2022hivt,
  title={HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction},
  author={Zhou, Zikang and Ye, Luyao and Wang, Jianping and Wu, Kui and Lu, Kejie},
  booktitle={CVPR},
  year={2022}
}

@inproceedings{zhao2021tnt,
  title={TNT: Target-driveN Trajectory Prediction},
  author={Zhao, Hang and Gao, Jiyang and Lan, Tian and Sun, Chen and Sapp, Ben and Varadarajan, Balakrishnan and Shen, Yue and Shen, Yi and Chai, Yuning and Schmid, Cordelia and others},
  booktitle={CoRL},
  year={2021}
}

@inproceedings{gu2021densetnt,
  title={DenseTNT: End-to-end Trajectory Prediction from Dense Goal Sets},
  author={Gu, Junru and Sun, Chen and Zhao, Hang},
  booktitle={ICCV},
  year={2021}
}

@article{v2xseq2023,
  title={V2X-Seq: A Large-Scale Sequential Dataset for Vehicle-Infrastructure Cooperative Perception and Forecasting},
  author={Yu, Haibao and Yang, Wenxian and Ruan, Hongzhi and others},
  booktitle={CVPR},
  year={2023}
}
```
