"""CivilComments-WILDS spurious correlation benchmark.

Binary classification: toxic (1) vs non-toxic (0).
Spurious attribute: mention of 8 demographic identities.
16 potentially overlapping groups (8 identities x toxic/non-toxic).

Source: CivilComments-WILDS (Koh et al., 2021, "WILDS: A Benchmark of
in-the-Wild Distribution Shifts").

Expects data at: {root}/civilcomments/
The dataset can be obtained via the WILDS package:
    pip install wilds
    python -c "from wilds import get_dataset; get_dataset('civilcomments', root_dir='./data')"

Or downloaded manually from:
    https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/
"""

import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


IDENTITY_COLS = [
    "male", "female", "LGBTQ", "christian", "muslim",
    "other_religions", "black", "white",
]


class CivilComments(Dataset):
    """CivilComments toxicity classification dataset.

    Returns (token_ids, y) where token_ids is a 1D int32 array of
    BERT token IDs (padded/truncated to max_length).

    Args:
        root: data root directory.
        split: "train", "val", or "test".
        max_length: maximum token sequence length. Default 300 (WILDS convention).
        tokenizer_name: HuggingFace tokenizer to use.
    """

    DATASET_DIR = "civilcomments"
    TOXICITY_THRESHOLD = 0.5

    def __init__(self, root="./data", split="train", max_length=300,
                 tokenizer_name="bert-base-uncased"):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        base_dir = os.path.join(root, self.DATASET_DIR)
        metadata_path = os.path.join(base_dir, "all_data_with_identities.csv")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"CivilComments data not found at {metadata_path}. "
                "Download via WILDS: pip install wilds && python -c "
                "\"from wilds import get_dataset; "
                "get_dataset('civilcomments', root_dir='./data')\""
            )

        # Check for cached tokenized data
        cache_path = os.path.join(base_dir, f"tokenized_{split}_{max_length}.npz")
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            self.token_ids = cached["token_ids"]
            self.y = cached["y"]
            self.group = cached["group"]
            self.y_bin = self.y
            return

        # Load and process raw data
        df = pd.read_csv(metadata_path)

        # Split mapping: WILDS uses article-level splits
        split_map = {"train": "train", "val": "val", "test": "test"}
        df = df[df["split"] == split_map[split]].reset_index(drop=True)

        # Binary toxicity label
        y = (df["toxicity"] >= self.TOXICITY_THRESHOLD).astype(np.int32).values

        # 16 overlapping groups: (identity_i, toxic) and (identity_i, non-toxic)
        # Group ID = identity_idx * 2 + toxic
        identity_matrix = np.zeros((len(df), 8), dtype=np.int32)
        for i, col in enumerate(IDENTITY_COLS):
            if col in df.columns:
                identity_matrix[:, i] = (df[col] >= 0.5).astype(np.int32).values

        # For worst-group eval, store per-sample group memberships
        # Primary group = first identity found (for single-group assignment)
        # Store full identity matrix for proper overlapping group eval
        self.identity_matrix = identity_matrix

        # Single group ID for compatibility (first identity * 2 + toxic)
        primary_identity = np.argmax(identity_matrix, axis=1)
        has_identity = identity_matrix.any(axis=1)
        # Samples with no identity get group = 16 (catch-all)
        group = np.where(has_identity, primary_identity * 2 + y, 16).astype(np.int32)

        # Tokenize
        try:
            from transformers import BertTokenizer
        except ImportError:
            raise ImportError(
                "CivilComments requires the 'transformers' package for tokenization. "
                "Install with: pip install transformers"
            )

        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        texts = df["comment_text"].tolist()

        print(f"  Tokenizing {len(texts)} comments (split={split})...")
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        token_ids = encoded["input_ids"].astype(np.int32)

        # Cache tokenized data
        os.makedirs(base_dir, exist_ok=True)
        np.savez(cache_path, token_ids=token_ids, y=y, group=group)
        print(f"  Cached tokenized data to {cache_path}")

        self.token_ids = token_ids
        self.y = y
        self.group = group
        self.y_bin = self.y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.y[idx]
