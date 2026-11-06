import sys, time, json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"D:\Year4\FA26\Capstone_Project\AIModerationContent")
sys.path.insert(0, str(PROJECT_ROOT))

from src.langid.detector import LanguageDetector  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"
TARGET_N = 29_934

df_vi = pd.read_csv(PROJECT_ROOT / "data" / "vinli" / "vinli_sentences.csv")
vi_texts = df_vi["text"].tolist()[:TARGET_N]

parts = []
for fn in ("wortschatz_split_train.csv", "wortschatz_split_val.csv", "wortschatz_split_test.csv"):
    parts.append(pd.read_csv(EVAL_DIR / "wortschatz" / fn)["sentence"].tolist())
en_texts = [t for part in parts for t in part][:TARGET_N]

all_texts = vi_texts + en_texts
detector = LanguageDetector()

# warm-up
for t in all_texts[:50]:
    detector.detect(t)

lat_ms = []
for t in all_texts:
    t0 = time.perf_counter()
    detector.detect(t)
    t1 = time.perf_counter()
    lat_ms.append((t1 - t0) * 1000)

lat_ms.sort()
n = len(lat_ms)


def pct(p):
    k = (p / 100) * (n - 1)
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return lat_ms[f]
    d = k - f
    return lat_ms[f] * (1 - d) + lat_ms[c] * d


summary = {
    "n": n,
    "vi_n": len(vi_texts),
    "en_n": len(en_texts),
    "mean_ms": round(sum(lat_ms) / n, 4),
    "p50_ms": round(pct(50), 4),
    "p95_ms": round(pct(95), 4),
    "p99_ms": round(pct(99), 4),
    "max_ms": round(max(lat_ms), 4),
    "min_ms": round(min(lat_ms), 4),
    "throughput_per_sec": round(n / (sum(lat_ms) / 1000), 2),
}

out = PROJECT_ROOT / "outputs" / "evaluation" / "langid" / "langid_vi_en_latency.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
