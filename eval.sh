python eval_classifier.py \
    --model_path "sameearif/deberta-v3-large-csb" \
    --dataset_name "sameearif/CSB-Classifier" \
    --threshold 0.5 \
    --device "mps" \
    --split "test"