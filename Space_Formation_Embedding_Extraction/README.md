# Step 2: Space Formulation & Embedding Extraction

## 2.1 Data-driven Aspect Space + LLM Teacher Scoring

We do **not** predefine aspects manually. Instead, we use an LLM to **derive a K-dimensional aspect space from the dataset itself** (data-driven taxonomy), then use the same LLM to **score each (description, review) pair** under this shared aspect space.

Specifically, `airbnb_aspect_space_llm.py` does the following:

- mines candidate phrases from descriptions + reviews
- asks the LLM to define **K = 10** aspect slots (names + definitions) from those phrases
- loads the existing taxonomy (`taxonomy.json`) if already built
- scores each selected row **once**, outputting two vectors in the **same numeric space [0,1]**:
  - `d`: description-side “claimed experience level” for each aspect
  - `r`: review-side “experienced level” for each aspect

The output is a **teacher-labeled CSV** with:

- `desc_S01_score … desc_S10_score`
- `rev_S01_score  … rev_S10_score`

This teacher CSV is later used to train a cheaper student model.

### Command (teacher scoring)

```bash
CUDA_VISIBLE_DEVICES=4 python airbnb_aspect_space_llm.py \
  --csv_path airbnb.csv \
  --listing_col listing_id \
  --desc_col description \
  --review_col review \
  --K 10 \
  --score_n 2000 \
  --score_unique_listing \
  --score_k_per_listing 4 \
  --score_seed 0 \
  --out_dir out_airbnb_k10
```

### Output
According to the actually num. of scored data, the output csv is:
``
out_airbnb_k10/teacher_scores_{num. of scored data}.csv
``


## 2.2 Train a Student Model (Distill the LLM Scoring)

LLM scoring is accurate but **too expensive** to run over the full dataset (13k+ rows).  
So we **distill** the LLM teacher into a smaller, fast model (DeBERTa) that learns the same scoring function.

Key idea:

- The teacher provides **two 10-d vectors** per row (desc and review), but both live in the **same shared 10-d aspect space**.
- We convert the teacher CSV into a regression dataset by **expanding each original row into 2 samples**:
  - input: `"[DESC]\n{description}"` → target: `desc_vec ∈ ℝ¹⁰`
  - input: `"[REV]\n{review}"`       → target: `rev_vec  ∈ ℝ¹⁰`

### Command (student training)

```bash
CUDA_VISIBLE_DEVICES=6 python train_student.py \
  --train_csv teacher_scores_2000.csv \
  --model_name microsoft/deberta-v3-large \
  --K 10 \
  --save_dir student_deberta_k10 \
  --max_length 512 \
  --train_bs 8 --eval_bs 16 \
  --epochs 2 \
  --lr 2e-5 \
  --bf16
```

### Output

The trained student model (and metadata) will be saved to:

```text
student_deberta_k10/
├── model.safetensors
├── student_meta.json
├── metrics_epoch.csv
└── curves_*.png
```

## 2.3 Evaluate the Student Model vs. Baseline

To verify that the student model is **actually learning a meaningful text → aspect mapping**, we compare it against a strong but simple baseline:

- **Baseline:** a constant predictor that always outputs the **training-set mean vector** (per dimension)
- **Student:** the fine-tuned DeBERTa regressor

Evaluation is performed **only on the held-out eval split**, not on the training data, to ensure fairness.

### What is evaluated

- Overall MAE and MSE
- Per-dimension MAE (10 dimensions)
- Relative improvement over baseline
- Bootstrap significance test:
  - Resamples the eval set with replacement
  - Estimates a 95% confidence interval for  
    \(\Delta = \text{MAE}_{\text{baseline}} - \text{MAE}_{\text{student}}\)

### Command (evaluation)

```bash
CUDA_VISIBLE_DEVICES=6 python eval_student_vs_baseline.py \
  --student_dir student_deberta_k10 \
  --teacher_csv teacher_scores_2000.csv \
  --batch_size 64 \
  --bootstrap 2000 \
  --device cuda
```

### Output

```text
=== Baseline (Constant train-mean) ===
eval_mae=0.114326  eval_mse=0.021949

=== Student (Loaded best model.safetensors) ===
eval_mae=0.058691  eval_mse=0.008147

=== Improvement ===
MAE_reduction = 0.055635  (relative 48.66%)
Per-dim wins: 10/10

=== Bootstrap significance ===
delta_mean=0.055625
95% CI = [0.051918, 0.059528]
P(delta > 0) = 1.0000
```
These results show that the student model significantly outperforms the constant-mean baseline across all dimensions, with strong statistical confidence.


## 2.4 Student Inference at Scale

After training and validating the student model, you can **replace the LLM scoring stage entirely** by running inference with the fine-tuned DeBERTa model on the full dataset (e.g., 13k Airbnb rows).

### Download Pretrained Student Weights

For convenience, we provide the fine-tuned **DeBERTa-v3-large** student model via Google Drive: [Link](https://drive.google.com/drive/folders/1GQpmQEm8Nbj3oiZuPV5DCeUVoWUtClGG?usp=sharing)

Download and place the files into the following directory structure:

```text
Project/
├── out_airbnb_k10/
├── student_deberta_k10/
│   ├── model.safetensors
│   ├── student_meta.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── spm.model
└── airbnb_aspect_space_llm.py
```

### Command (student inference)

Once the weights are in place, run the student inference script to score **all descriptions and reviews** in the base CSV:

```bash
CUDA_VISIBLE_DEVICES=6 python infer_student.py \
  --student_dir student_deberta_k10 \
  --csv_path airbnb.csv \
  --out_csv student_scores_13k.csv \
  --desc_col description \
  --review_col review \
  --row_id_col row_idx \
  --batch_size 32 \
  --clamp \
  --log_every 200
```

This produces a CSV that **preserves all original columns from the base CSV**
(e.g., listing metadata, description, review, etc.), and **appends 20 new score columns**
predicted by the student model.

### Output (appended)

```text
desc_S01_score, ..., desc_S10_score,
rev_S01_score,  ..., rev_S10_score
```

