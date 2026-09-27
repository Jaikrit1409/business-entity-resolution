# Business Entity Resolution

This project contains the data, source code, experiments, and model artifacts for a business entity resolution pipeline.

## Project structure

- `data/train/` - labeled training data and ground truth
- `data/test/` - unlabeled test data
- `src/` - data processing, feature engineering, matching, and model code
- `notebooks/` - exploratory analysis and validation notebooks
- `experiments/` - experiment tracking and configuration files
- `models/` - trained model artifacts
- `output/` - predictions and evaluation outputs
- `tests/` - unit and integration tests

Generated model artifacts are written to `models/`, and generated TSV files are
written to `output/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Notes

- Add source-specific training and test files under the corresponding folders.
- Keep all scripts reusable and version-controlled.
