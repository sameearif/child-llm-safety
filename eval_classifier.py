import argparse
import logging
import sys
import os
import string
import json
import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments
)
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. SETUP LOGGING ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier with custom thresholds.")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--dataset_name", type=str, default="sameearif/CSB_Classifier", help="Hugging Face dataset ID")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for the positive class (Accepted). Default 0.5")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on (test, validation, train)")
    parser.add_argument("--output_file", type=str, default="safety_risks.json", help="Filename to save false positives")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device to use.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Starting evaluation with threshold: {args.threshold} on device: {args.device}")

    # --- 2. SETUP DEVICE ---
    use_cpu = False
    use_mps = False
    
    if args.device == "cpu":
        use_cpu = True
    elif args.device == "mps":
        use_mps = True
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to auto-detection.")
    
    training_args = TrainingArguments(
        output_dir="./eval_results_temp", 
        per_device_eval_batch_size=args.batch_size,
        use_cpu=use_cpu,
        use_mps_device=use_mps,
        report_to="none", 
    )

    # --- 3. LOAD MODEL & TOKENIZER ---
    try:
        logger.info(f"Loading model from: {args.model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # --- 4. LOAD DATASET ---
    logger.info(f"Loading dataset: {args.dataset_name} [{args.split}]")
    try:
        dataset = load_dataset(args.dataset_name, split=args.split)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # --- 5. PREPROCESSING ---
    logger.info("Applying text cleaning (lowercase + punctuation removal)...")

    def clean_text(example):
        text_column = "text" if "text" in example else "sentence"
        text = example[text_column]
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        example[text_column] = text
        return example

    # We map this to clean the text, but keep all other columns!
    dataset_clean = dataset.map(clean_text)

    def preprocess_function(examples):
        text_column = "text" if "text" in examples else "sentence"
        return tokenizer(
            examples[text_column], 
            truncation=True, 
            padding="max_length", 
            max_length=tokenizer.model_max_length if tokenizer.model_max_length < 10000 else 512
        )

    tokenized_dataset = dataset_clean.map(preprocess_function, batched=True)

    # --- 6. INFERENCE ---
    logger.info("Running inference...")
    
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer,
        args=training_args
    )
    
    outputs = trainer.predict(tokenized_dataset)
    logits = outputs.predictions
    labels = outputs.label_ids

    # --- 7. APPLY THRESHOLD LOGIC ---
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    positive_probs = probs[:, 1]
    predictions = (positive_probs >= args.threshold).astype(int)

    # --- 8. METRICS ---
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    acc = acc_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    # --- 9. REPORT ---
    print("\n" + "="*50)
    print(f"EVALUATION REPORT (Threshold: {args.threshold})")
    print("="*50)
    print(f"Device Used:    {training_args.device}")
    print(f"Dataset Size:   {len(dataset)}")
    print(f"Accuracy:       {acc:.4f}")
    print(f"F1 Score:       {f1:.4f}")
    print("-" * 50)
    print(f"True Positives (Safe & Accepted):      {tp}")
    print(f"True Negatives (Unsafe & Rejected):    {tn}")
    print("-" * 50)
    print(f"⚠️  False Positives (Unsafe allowed):   {fp}  <-- SAFETY RISK")
    print(f"📉 False Negatives (Safe blocked):     {fn}  <-- USABILITY RISK")
    print("-" * 50)
    
    print("\nDetailed Report:")
    print(classification_report(labels, predictions, target_names=["Rejected (0)", "Accepted (1)"]))

    if fp == 0:
        print("✅ SUCCESS: Zero False Positives. The model is safe at this threshold.")
    else:
        print(f"❌ WARNING: {fp} Unsafe items were allowed through. Consider RAISING the threshold.")

    # --- 10. SAVE SAFETY RISKS (FULL DATA) ---
    logger.info("Extracting False Positive examples with FULL metadata...")
    
    safety_risks = []
    
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        if label == 0 and pred == 1:
            # 1. Grab the ENTIRE dictionary for this row (includes source, debug_scores, etc.)
            # We copy it to avoid modifying the dataset object in memory
            full_entry = dataset[i].copy()
            
            # 2. Add our inference metadata to it
            full_entry.update({
                "ANALYSIS_predicted_label": 1,
                "ANALYSIS_ground_truth": 0,
                "ANALYSIS_confidence_score": float(positive_probs[i]),
                "ANALYSIS_threshold": args.threshold,
                "ANALYSIS_error_type": "False Positive"
            })
            
            safety_risks.append(full_entry)
            
    # Save to JSON
    if safety_risks:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(safety_risks, f, indent=4, ensure_ascii=False)
        print(f"\n⚠️  SAFETY REPORT: Saved {len(safety_risks)} unsafe failures (with full metadata) to '{args.output_file}'")
    else:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
        print(f"\n✅ SAFETY REPORT: No safety risks found. '{args.output_file}' is empty.")

if __name__ == "__main__":
    main()