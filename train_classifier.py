import argparse
import logging
import os
import sys
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a text classifier on the CSB dataset.")
    
    # Model & Data Arguments
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base", help="Pretrained model alias")
    parser.add_argument("--dataset_name", type=str, default="sameearif/CSB_Classifier", help="Hugging Face dataset ID")
    parser.add_argument("--output_dir", type=str, default="./results", help="Where to store model checkpoints")
    
    # Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and eval")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    
    # Auto-calculate max_length if not provided
    parser.add_argument("--max_length", type=int, default=None, help="Max token length. If None, calculates from data.")
    
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Hub Arguments
    parser.add_argument("--push_to_hub", action="store_true", help="Push final model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Repository name if pushing to hub")

    return parser.parse_args()

def compute_metrics(eval_pred):
    """
    Compute accuracy and F1 score for evaluation.
    """
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}

def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    logger.info(f"Starting training with parameters: {args}")

    # --- 1. Load Dataset ---
    logger.info(f"Loading dataset: {args.dataset_name}")
    try:
        dataset = load_dataset(args.dataset_name)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # Ensure validation split exists
    if "validation" not in dataset:
        logger.info("No validation split found. Splitting training data...")
        split = dataset["train"].train_test_split(test_size=0.1, seed=args.seed)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # --- 2. Prepare Tokenizer ---
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # --- 3. Calculate Max Length (if None) ---
    if args.max_length is None:
        logger.info("max_length is None. Calculating optimal length from dataset...")
        
        sample = dataset["train"][0]
        text_column = "text" if "text" in sample else "sentence"
        
        # Fast token count estimation
        def get_token_len(example):
            return {"len": len(tokenizer(example[text_column], truncation=False)["input_ids"])}
        
        logger.info("Scanning training data for sequence lengths...")
        train_lengths = dataset["train"].map(
            get_token_len, 
            batched=False, 
            remove_columns=dataset["train"].column_names
        )
        
        # args.max_length = max(train_lengths["len"])
        args.max_length = 512

        logger.info(f"Set max_length to {args.max_length}")

    def preprocess_function(examples):
        text_column = "text" if "text" in examples else "sentence"
        return tokenizer(
            examples[text_column], 
            truncation=True, 
            padding="max_length", 
            max_length=args.max_length
        )

    logger.info("Tokenizing dataset...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # --- 4. Prepare Model ---
    # Auto-detect labels
    label_list = dataset["train"].unique("label")
    label_list.sort()
    num_labels = len(label_list)
    
    # Map Labels (Adjust if you have different classes)
    id2label = {0: "Rejected", 1: "Accepted"}
    label2id = {"Rejected": 0, "Accepted": 1}
    
    # Fallback for multi-class
    if num_labels != 2:
        id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
        label2id = {v: k for k, v in id2label.items()}

    logger.info(f"Loading model: {args.model_name} with {num_labels} labels")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # --- 5. Training Arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",  
        save_strategy="epoch",        
        load_best_model_at_end=True,
        metric_for_best_model="f1",   
        save_total_limit=2,           
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        report_to="none", 
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- 6. Train ---
    logger.info("Starting training...")
    trainer.train()

    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")

    # --- 7. Save & Push Best Model ---
    
    # Save Best Model Locally to specific folder
    model_slug = args.model_name.split("/")[-1]
    best_model_dir = os.path.join(args.output_dir, f"{model_slug}_best")
    
    logger.info(f"Saving best model to local directory: {best_model_dir}")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    # Push Best Model to Hub
    if args.push_to_hub:
        logger.info("Pushing BEST model to Hugging Face Hub...")
        # Since load_best_model_at_end=True, trainer.push_to_hub() uses the best weights currently in memory
        trainer.push_to_hub(commit_message="Training complete: Pushing Best Model")
        logger.info(f"Successfully pushed model to: {args.hub_model_id}")

if __name__ == "__main__":
    main()