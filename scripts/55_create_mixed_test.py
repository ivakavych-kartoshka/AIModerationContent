import pandas as pd
from pathlib import Path

# Load files
vi_test = pd.read_csv(r'D:\Year4\FA26\Capstone_Project\AIModerationContent\data\splits\test.csv')
en_test = pd.read_csv(r'D:\Year4\FA26\Capstone_Project\AIModerationContent\data\splits_en\test.csv')

# Add language label
vi_test['true_lang'] = 'vi'
en_test['true_lang'] = 'en'

# Combine
combined = pd.concat([vi_test, en_test], ignore_index=True)

# Shuffle
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
out_path = Path(r'D:\Year4\FA26\Capstone_Project\AIModerationContent\outputs\evaluation\combined_mixed_test.csv')
out_path.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(out_path, index=False, encoding='utf-8')
print(f"Saved {len(combined)} combined samples to {out_path}")
