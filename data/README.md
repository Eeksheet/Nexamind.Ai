# Data

| Directory | Contents | Tracked in git? |
|---|---|---|
| `raw/` | HotpotQA source files (HF parquet shard `distractor/validation-00000-of-00001.parquet`, or official `hotpot_dev_distractor_v1.json`) | no |
| `processed/` | Normalised JSON cache (`hotpotqa_distractor_validation_full.json`), embedding cache, LLM cache | no |

## Obtaining HotpotQA

Automatic (default): the first call to `src.datasets.load_dataset` downloads the
public HuggingFace mirror `hotpotqa/hotpot_qa` (distractor / validation, 7 405
questions, ~45 MB) into `data/raw/`.

```bash
python -c "from src.datasets import load_dataset; from src.utils import load_config; print(load_dataset(load_config().dataset).label)"
```

Manual: place `hotpot_dev_distractor_v1.json` from https://hotpotqa.github.io/ in `data/raw/`
— it is preferred over the parquet mirror when present.

## Synthetic data

`src.datasets.synthetic_multihop(n, seed)` generates deterministic 2-hop bridge
questions with distractor paragraphs. It is used by the test-suite and by
`--synthetic` smoke runs. Every dataset object carries `kind = real | synthetic`
and every results table/manifest records it; synthetic numbers are never
benchmark numbers.
