# Brain v0.5

Unified package: core LM + Tools + Agents + Knowledge + Reasoning + Memory +
Tokenizers + Datasets + Training + GPT-2 family + v5 sparse/deep/wide configs.

```python
from Models import create_model
m = create_model("o-mini")       # 33.2M
m = create_model("gpt2-small")   # classic GPT-2 Small recipe
m = create_model("v5")           # brain-v5-emergent architecture
m = create_model("v5-96")        # 96 layers, 96 heads (research)
```
