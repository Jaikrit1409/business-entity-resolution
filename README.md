# Business Entity Resolution — Amazon ML Challenge 2026

An end-to-end, deterministic machine learning pipeline for large-scale multi-source business entity resolution. Resolves corporate identities across three noisy sources without common identifiers, optimizing for **Entity-Level Macro $F_{0.5}$**.

---

## 1. Project Directory Structure

```text
├── dataset/
│   ├── train/                 # train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
│   └── test/                  # test_source1.tsv, test_source2.tsv, test_source3.tsv
├── src/
│   ├── blocking.py            # Scalable inverted index & DuckDB candidate generation
│   ├── preprocess.py          # Conservative text normalization (open-set country, legal suffix stripping)
│   ├── feature_engineering.py # 33 pairwise features (RapidFuzz fuzzy metrics, address numerics, interactions)
│   ├── pair_generator.py      # Leakage-free Source 1 group splitting & candidate-only pair labeling
│   ├── train_model.py         # Precision-tuned LightGBM training & 91-step F0.5 threshold selection
│   ├── predict.py             # Candidate probability scoring and match selection
│   ├── submission.py          # Competition TSV schema formatting
│   ├── data_loader.py         # TSV streaming and dataset loaders
│   └── evaluate.py            # Macro F0.5 evaluation metrics
├── tests/                     # Unit and regression test suite
├── utils/
│   └── validate_submission.py # Official stdlib competition submission validator
├── output/
│   ├── matching_results.tsv   # Scored matches uploaded to the portal
│   └── candidate_pairs.tsv    # Candidate set emitted by blocking
├── Documentation_template.md  # Methodology write-up
├── package_submission.py      # Automated submission zip generator
├── requirements.txt           # Pinned dependencies
└── pytest.ini                 # Pytest configuration
```

---

## 2. Environment Setup

Python 3.10 to 3.12 is recommended.

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows

# 2. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt pytest
```

---

## 3. Running the Test Suite

Run the full suite of 27 unit, integration, and mathematical regression tests:

```bash
pytest -v
```

All tests execute against self-contained synthetic fixtures in seconds without requiring challenge datasets.

---

## 4. End-to-End Pipeline Reproduction

### Step 4.1: Candidate Generation (Blocking)
Produces `output/candidate_pairs.tsv`:
```bash
python3 -c "
from pathlib import Path
from src.blocking import MultiStrategyBlocker, BlockingConfig
from src.data_loader import load_records_from_tsv
from src.submission import write_candidate_pairs

s1 = load_records_from_tsv('dataset/test/test_source1.tsv')
s2 = load_records_from_tsv('dataset/test/test_source2.tsv')
s3 = load_records_from_tsv('dataset/test/test_source3.tsv')

blocker = MultiStrategyBlocker(BlockingConfig())
candidates = blocker.generate(s1, s2, s3)
write_candidate_pairs(candidates.combined(), 'output/candidate_pairs.tsv')
print('Candidate generation complete.')
"
```

### Step 4.2: Model Training & Threshold Selection
Trains precision-conscious LightGBM and tunes decision threshold $\tau^*$ on validation entities:
```bash
python3 -c "
from src.data_loader import load_records_from_tsv
from src.train_model import train_baseline_model

s1 = load_records_from_tsv('dataset/train/train_source1.tsv')
s2 = load_records_from_tsv('dataset/train/train_source2.tsv')
s3 = load_records_from_tsv('dataset/train/train_source3.tsv')
truth = load_records_from_tsv('dataset/train/train_ground_truth.tsv')

result = train_baseline_model(
    source1_records=s1,
    source2_records=s2,
    source3_records=s3,
    ground_truth_records=truth,
    output_dir='models',
    scale_pos_weight=1.0,
)
print(f'Model trained. Selected F0.5 Threshold: {result.selected_threshold}')
"
```

### Step 4.3: Scoring & Generating `matching_results.tsv`
```bash
python3 -c "
from src.data_loader import load_records_from_tsv
from src.predict import score_candidates, select_matches
from src.submission import write_matching_results
import json

s1 = load_records_from_tsv('dataset/test/test_source1.tsv')
s2 = load_records_from_tsv('dataset/test/test_source2.tsv')
s3 = load_records_from_tsv('dataset/test/test_source3.tsv')

with open('models/baseline_config.json') as f:
    config = json.load(f)
threshold = config['selected_threshold']

matrix, probs = score_candidates(s1, s2, s3, 'models/lightgbm.txt', 'models/baseline_config.json')
matches = select_matches(matrix, probs, threshold=threshold)

s1_ids = [r['entity_id'] for r in s1]
write_matching_results(s1_ids, matches, 'output/matching_results.tsv')
print('Matching results written.')
"
```

---

## 5. Pre-Submission Validation

Before submitting to the portal, validate both TSVs against competition constraints:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

---

## 6. Packaging for Final Submission

Generate the required `<team_name>_submission.zip`:

```bash
python3 package_submission.py --team-name <your_team_name>
```

---

## 7. Model Licensing & Constraints
- **License**: LightGBM is MIT licensed. RapidFuzz is MIT licensed. DuckDB is MIT licensed.
- **Model Size**: Parameter footprint < 10 MB (well below the 8 Billion parameter competition ceiling).
- **External Data**: Zero external data lookups, zero geocoding APIs, zero web scraping.
