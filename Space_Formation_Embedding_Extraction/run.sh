CUDA_VISIBLE_DEVICES=4 python airbnb_aspect_space_llm.py \
  --csv_path airbnb.csv \
  --listing_col listing_id \
  --desc_col description \
  --review_col review \
  --K 10 \
  --score_n 4000 \
  --score_unique_listing \
  --score_k_per_listing 4 \
  --score_seed 0 \
  --out_dir out_airbnb_k10


CUDA_VISIBLE_DEVICES=4 python train_student.py \
  --train_csv out_airbnb_k10/teacher_scores_2000.csv \
  --save_dir student_deberta_al \
  --model_name microsoft/deberta-v3-large \
  --K 10 --max_length 512 \
  --train_bs 8 --eval_bs 16 --epochs 20 \
  --lr 2e-5 --weight_decay 0.01 --warmup_ratio 0.03 \
  --fp16


CUDA_VISIBLE_DEVICES=4 python infer_student.py \
  --student_dir student_deberta_al \
  --csv_path airbnb-amsterdam.csv \
  --out_csv amsterdam_student_scores_joint.csv \
  --desc_col description --review_col review \
  --batch_size 32 --clamp


CUDA_VISIBLE_DEVICES=5 python eval_student_vs_baseline.py \
  --student_dir student_deberta_k10 \
  --teacher_csv teacher_scores_2000.csv \
  --batch_size 64 \
  --bootstrap 2000 \
  --device cuda
