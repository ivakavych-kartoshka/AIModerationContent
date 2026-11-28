import pandas as pd
from pathlib import Path
from src.langid.detector import LanguageDetector

df = pd.read_csv('outputs/evaluation/combined_mixed_test.csv')
detector = LanguageDetector()

correct = 0
total = len(df)
confusion = {}

results = []
for idx, row in df.iterrows():
    text = row['text']
    true_lang = row['true_lang']
    res = detector.detect(text)
    pred_lang = res.lang
    
    # Map 'mix' or 'other' appropriately for evaluation if needed, but let's see raw matches first
    if pred_lang == true_lang:
        correct += 1
    
    confusion.setdefault(true_lang, {}).setdefault(pred_lang, 0)
    confusion[true_lang][pred_lang] += 1

print(f"Total samples: {total}")
print(f"Exact matches (pred_lang == true_lang): {correct} ({correct/total*100:.2f}%)")
print("\nConfusion Matrix (true_lang -> pred_counts):")
for t_lang, preds in confusion.items():
    print(f"  True {t_lang}: {preds}")
