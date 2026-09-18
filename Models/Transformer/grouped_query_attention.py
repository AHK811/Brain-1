from __future__ import annotations
# Re-export / bridge to brain GQA attention
def get_gqa_attention_class():
    from brain.model.transformer.attention import GroupedQueryAttention
    return GroupedQueryAttention
