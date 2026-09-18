# Brain v0.5 UNIFIED — correctness (v0.4 fixed) + capabilities (v0.5)

## Merge principle
- **From v0.4 upgraded:** cached-generation `is_causal` fix, attention_mask restore,
  MHA validation, RoPE device/dtype, checkpoint effective_theta, deep64, phase6 tests,
  honest registry, 250m/500m param locks (qk_norm off).
- **From v0.5:** book corpus pipeline (clean/dedup/pack/memmap), Gutenberg importer,
  Alternating Sparse Attention, v5-emergent / v5-96 configs, capability scaffolds.

## Correctness
| Item | Status |
|------|--------|
| Cached decode `is_causal` | Fixed (v0.4) |
| attention_mask training path | Restored (v0.4) |
| o-mini params | 33,236,352 |
| deep64 | 839M tier present |
| Book pipeline | Present + tested in source zip |

## New capability
| Item | Status |
|------|--------|
| AlternatingSparseAttention | Wired |
| v5-emergent / v5-96 configs | Present |
| Book corpus CLI | scripts/build_book_corpus.py |

## Create models
```python
from Models import create_model
create_model("o-mini")
create_model("deep64")
create_model("v5")
create_model("gpt2-small")
```
