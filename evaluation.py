import os
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

nltk.download('punkt')

def load_reference_translations(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        references = file.readlines()
    return [ref.strip() for ref in references]

def evaluate_bleu(reference, candidate):
    smoothie = SmoothingFunction().method4
    reference_tokens = nltk.word_tokenize(reference)
    candidate_tokens = nltk.word_tokenize(candidate)
    score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
    return score

def evaluate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

def evaluate_translations(reference_file, generated_translations):
    references = load_reference_translations(reference_file)
    
    if len(references) != len(generated_translations):
        raise ValueError("The number of references and generated translations must be the same.")
    
    bleu_scores = []
    rouge_scores = {'rouge1': [], 'rougeL': []}
    
    for ref, gen in zip(references, generated_translations):
        bleu_score = evaluate_bleu(ref, gen)
        rouge_score = evaluate_rouge(ref, gen)
        
        bleu_scores.append(bleu_score)
        rouge_scores['rouge1'].append(rouge_score['rouge1'].fmeasure)
        rouge_scores['rougeL'].append(rouge_score['rougeL'].fmeasure)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge1 = sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1'])
    avg_rougeL = sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])
    
    return {
        'avg_bleu': avg_bleu,
        'avg_rouge1': avg_rouge1,
        'avg_rougeL': avg_rougeL,
    }
