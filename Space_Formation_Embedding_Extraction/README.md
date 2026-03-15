# Step 2: Space Formation & Embedding Extraction

This folder builds a shared aspect space, produces LLM teacher scores, trains a student model, and runs inference/evaluation.

## Pipeline

1. Build aspect space + teacher labels
2. Train student regressor
3. Run student inference on full CSV
4. Evaluate student vs baseline

## Main scripts and commands

### 1) LLM teacher scoring

```bash
CUDA_VISIBLE_DEVICES=4 python airbnb_aspect_space_llm.py \
  --csv_path airbnb-al.csv \
  --listing_col listing_id \
  --desc_col description \
  --review_col review \
  --K 10 \
  --score_n 4000 \
  --score_unique_listing \
  --score_k_per_listing 4 \
  --score_seed 0 \
  --out_dir out_airbnb_k10
```

Output includes:
- `out_airbnb_k10/taxonomy.json`
- `out_airbnb_k10/teacher_scores_*.csv`

### 2) Train student model (shared 10-d)

```bash
CUDA_VISIBLE_DEVICES=4 python train_student.py \
  --train_csv out_airbnb_k10/teacher_scores_2000.csv \
  --save_dir student_deberta_al \
  --model_name microsoft/deberta-v3-large \
  --K 10 --max_length 512 \
  --train_bs 8 --eval_bs 16 --epochs 20 \
  --lr 2e-5 --weight_decay 0.01 --warmup_ratio 0.03 \
  --fp16
```

### 3) Inference with trained student

```bash
CUDA_VISIBLE_DEVICES=4 python infer_student.py \
  --student_dir student_deberta_al \
  --csv_path airbnb-al.csv \
  --out_csv student_scores_al.csv \
  --desc_col description --review_col review \
  --batch_size 32 --clamp
```

### 4) Evaluate student vs baseline

```bash
CUDA_VISIBLE_DEVICES=5 python eval_student_vs_baseline.py \
  --student_dir student_deberta_al \
  --teacher_csv out_airbnb_k10/teacher_scores_2000.csv \
  --batch_size 64 \
  --bootstrap 2000 \
  --device cuda
```

## Note

`run.sh` contains working command templates used in this project. Update file paths (`--csv_path`, `--train_csv`, `--student_dir`) to match the dataset split you are running.
