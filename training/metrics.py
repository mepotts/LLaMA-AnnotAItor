from typing import Dict, Any
import evaluate
from datasets import load_metric
import numpy as np
import torch
from transformers import EvalPrediction
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Download required NLTK data
try:
    nltk.download('punkt')
except:
    pass

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute evaluation metrics for model predictions
    """
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # Convert logits to predictions if necessary
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Decode predictions and labels
    tokenizer = eval_pred.tokenizer
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Initialize metrics
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate metrics
    results = {}
    
    # ROUGE scores
    rouge_scores = [rouge.score(pred, label) for pred, label in zip(decoded_preds, decoded_labels)]
    results['rouge1'] = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
    results['rouge2'] = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
    results['rougeL'] = np.mean([score['rougeL'].fmeasure for score in rouge_scores])
    
    # BLEU score
    def calculate_bleu(pred, label):
        pred_tokens = nltk.word_tokenize(pred.lower())
        label_tokens = nltk.word_tokenize(label.lower())
        return sentence_bleu([label_tokens], pred_tokens)
    
    bleu_scores = [calculate_bleu(pred, label) 
                   for pred, label in zip(decoded_preds, decoded_labels)]
    results['bleu'] = np.mean(bleu_scores)
    
    # Length metrics
    pred_lens = [len(nltk.word_tokenize(pred)) for pred in decoded_preds]
    label_lens = [len(nltk.word_tokenize(label)) for label in decoded_labels]
    results['gen_len'] = np.mean(pred_lens)
    results['ref_len'] = np.mean(label_lens)
    results['len_ratio'] = np.mean([p/l if l > 0 else 0 
                                   for p, l in zip(pred_lens, label_lens)])
    
    return results

def validate_outputs(model, tokenizer, validation_data, config):
    """
    Validate model outputs on a separate validation set
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for item in validation_data:
            # Generate prediction
            inputs = tokenizer(
                item['prompt'],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_length
            ).to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.inference.max_new_tokens,
                temperature=config.inference.temperature,
                top_p=config.inference.top_p,
                top_k=config.inference.top_k,
                repetition_penalty=config.inference.repetition_penalty,
                do_sample=True
            )
            
            # Decode prediction
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate metrics for this sample
            sample_metrics = {
                'prompt': item['prompt'],
                'prediction': pred_text,
                'reference': item['completion'],
                'metrics': compute_metrics(EvalPrediction(
                    predictions=outputs,
                    label_ids=tokenizer(
                        item['completion'],
                        return_tensors="pt"
                    ).input_ids
                ))
            }
            
            results.append(sample_metrics)
    
    return results