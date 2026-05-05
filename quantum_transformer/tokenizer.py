"""
Professional Byte-Pair Encoding (BPE) Tokenizer.

Uses the Hugging Face 'tokenizers' library to provide world-class
sub-word tokenization. This allows the model to process 
entire words or syllables as single tokens, drastically 
improving fluency and training efficiency.
"""

import os
import json
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

class BPETokenizer:
    """
    World-class BPE Tokenizer.
    Groups common byte sequences into single tokens.
    """

    def __init__(self, vocab_size: int = 8192):
        self.tokenizer = None
        self.vocab_size = vocab_size

    def train(self, files: list, vocab_size: int = None):
        """Train a BPE tokenizer on a list of text files."""
        if vocab_size:
            self.vocab_size = vocab_size

        # Create a Byte-level BPE model (Standard for Llama/GPT)
        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )

        self.tokenizer.train(files, trainer)

    def save_pretrained(self, path: str):
        """Save tokenizer files in Hugging Face style."""
        os.makedirs(path, exist_ok=True)
        # Save the full tokenizer artifact
        self.tokenizer.save(os.path.join(path, "tokenizer.json"))
        # Save a human-readable vocab file
        vocab = self.tokenizer.get_vocab()
        with open(os.path.join(path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)

    def load_pretrained(self, path: str):
        """Load tokenizer from directory."""
        tokenizer_path = os.path.join(path, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded!")
        return self.tokenizer.encode(text).ids

    def decode(self, tokens: list) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)

    def encode_batch(self, texts: list) -> list:
        return [self.encode(t) for t in texts]


# Keep the old ByteTokenizer for legacy/fallback support
class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.vocab = {str(i): i for i in range(256)}

    def save_pretrained(self, path: str):
        import json
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "vocab.json"), "w") as f:
            json.dump(self.vocab, f, indent=2)

    def encode(self, text: str) -> list:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return bytes(tokens).decode("utf-8", errors="replace")

    def encode_batch(self, texts: list) -> list:
        return [self.encode(t) for t in texts]
