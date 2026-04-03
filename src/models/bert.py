"""Flax BERT wrapper for text classification.

Uses HuggingFace's FlaxBertModel for the CivilComments benchmark.
Requires: pip install transformers[flax]

Follows the project convention:
    __call__(self, x, train=True) -> (logits, {"z": [CLS] embedding})
where x is (B, seq_len) int32 token IDs.
"""

import jax.numpy as jnp
from flax import linen as nn


class BertClassifier(nn.Module):
    """BERT-based binary classifier.

    Loads a pretrained BERT model and adds a classification head.
    The bottleneck z is the [CLS] token embedding (768-d for bert-base).

    Args:
        num_classes: number of output logits.
        bert_name: HuggingFace model name.
        hidden_dim: hidden dimension of the BERT model (768 for base).
    """
    num_classes: int = 2
    bert_name: str = "bert-base-uncased"
    hidden_dim: int = 768

    def setup(self):
        try:
            from transformers import FlaxBertModel
        except ImportError:
            raise ImportError(
                "BertClassifier requires the 'transformers' package with Flax support. "
                "Install with: pip install transformers[flax]"
            )
        self.bert = FlaxBertModel.from_pretrained(self.bert_name)
        self.classifier = nn.Dense(self.num_classes)

    def __call__(self, x, train: bool = True):
        """Forward pass.

        Args:
            x: (B, seq_len) int32 token IDs.
            train: whether in training mode (affects dropout).

        Returns:
            (logits, {"z": cls_embedding}) where:
                logits: (B, num_classes)
                cls_embedding: (B, hidden_dim) [CLS] token representation.
        """
        attention_mask = (x != 0).astype(jnp.int32)
        outputs = self.bert(
            input_ids=x,
            attention_mask=attention_mask,
            deterministic=not train,
        )
        # [CLS] token is at position 0
        z = outputs.last_hidden_state[:, 0, :]  # (B, hidden_dim)
        logits = self.classifier(z)
        return logits, {"z": z}
