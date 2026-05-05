"""
Quantum Transformer Language Model.

Complete autoregressive language model built from quantum-inspired
transformer blocks. The architecture:

    Input IDs → Embedding + Position → [Quantum Transformer Block] × N
    → LayerNorm → Output Head → Logits

Supports save/load for pretrained checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

from quantum_transformer.config import QuantumTransformerConfig
from quantum_transformer.layers import QuantumTransformerBlock


class QuantumTransformerLM(nn.Module):
    """
    Quantum Transformer Language Model.

    A causal (autoregressive) language model where the transformer
    blocks use quantum-inspired attention and feed-forward layers.
    """

    def __init__(self, config: QuantumTransformerConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_context_length, config.hidden_dim)
        self.embed_dropout = nn.Dropout(config.dropout)

        # Stack of quantum transformer blocks
        self.blocks = nn.ModuleList([
            QuantumTransformerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                num_qubits=config.num_qubits,
                ffn_dim=config.ffn_dim,
                circuit_depth=config.circuit_depth,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # Final layer norm and output head
        self.ln_final = nn.LayerNorm(config.hidden_dim)
        self.output_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying: share embedding weights with output head
        self.output_head.weight = self.token_embedding.weight

        # Causal mask (registered as buffer so it moves with the model)
        mask = torch.tril(torch.ones(config.max_context_length,
                                      config.max_context_length))
        self.register_buffer("causal_mask", mask)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor,
                targets: torch.Tensor = None,
                past_key_values: list = None):
        """
        Args:
            input_ids: (batch, seq_len) long tensor of token IDs
            targets: (batch, seq_len) optional targets for loss computation
            past_key_values: Optional list of (K, V) tuples from previous turns

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar loss if targets provided, else None
            new_key_values: List of (K, V) for next turn
        """
        B, S = input_ids.shape
        device = input_ids.device

        prev_len = 0
        if past_key_values is not None:
            prev_len = past_key_values[0][0].shape[2] 
        
        S_total = prev_len + S

        # Context Window Management
        if S_total > self.config.max_context_length:
            offset = S_total - self.config.max_context_length
            prev_len = max(0, prev_len - offset)
            # Re-calculate positions for the last 'S' tokens relative to the NEW window
            positions = torch.arange(prev_len, prev_len + S, device=device).unsqueeze(0)
            if past_key_values is not None:
                past_key_values = [(k[:, :, offset:], v[:, :, offset:]) for k, v in past_key_values]
        else:
            positions = torch.arange(prev_len, S_total, device=device).unsqueeze(0)  # (1, S)
        
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)

        # Causal mask for this sequence length
        # In KV cache mode, the mask is a (S, S_total) matrix where S is usually 1
        mask = self.causal_mask[prev_len:S_total, :S_total]

        # Pass through quantum transformer blocks with KV Cache support
        new_key_values = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, kv = block(x, mask, past_key_value=past_kv)
            new_key_values.append(kv)

        # Final norm and project to vocabulary
        x = self.ln_final(x)
        logits = self.output_head(x)  # (B, S, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0  # Ignore padding tokens in InstructionDataset
            )

        return logits, loss, new_key_values

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 200,
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9,
                 repetition_penalty: float = 1.0, 
                 stop_token_ids: list = None,
                 return_probs: bool = False) -> torch.Tensor:
        """
        Autoregressive generation with KV Caching and Entropy-based Confidence.
        """
        self.eval()
        device = input_ids.device
        
        curr_ids = input_ids[:, -self.config.max_context_length:]
        logits, _, past_key_values = self(curr_ids)
        
        token_probs = []

        for step in range(max_new_tokens):
            # Apply temperature scaling
            next_logits = logits[:, -1, :] / max(temperature, 1e-4)

            # ── Standard Repetition Penalty (Localized) ──
            if repetition_penalty != 1.0 and len(input_ids[0]) > 0:
                # Apply penalty to recent tokens to prevent permanent grammatical degradation
                tokens_seen = list(set(input_ids[0].tolist()[-128:]))
                score = torch.gather(next_logits, 1, torch.tensor([tokens_seen], device=device))
                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                next_logits.scatter_(1, torch.tensor([tokens_seen], device=device), score)

            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # ── Structural Confidence Guard ──
            ent = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
            certainty = (1.0 - (ent / math.log(probs.size(-1)))) ** 2
            
            # Repetition Punishment: If we are repeating, confidence is 0
            last_tokens = input_ids[0].tolist()[-5:]
            if len(last_tokens) >= 4 and len(set(last_tokens)) <= 2:
                certainty = 0.0

            step_confidence = (probs[0, next_token.item()].item() * 0.3) + (certainty * 0.7)
            token_probs.append(step_confidence)
            
            token_id = next_token.item()
            if stop_token_ids and token_id in stop_token_ids: break

            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Correct KV Cache sliding window for absolute positional embeddings
            if past_key_values and past_key_values[0][0].shape[2] >= self.config.max_context_length:
                # KV cache is full. Shifting absolute embeddings corrupts them. 
                # Instead, recompute cleanly over the current context window limit.
                curr_ids = input_ids[:, -self.config.max_context_length:]
                logits, _, past_key_values = self(curr_ids)
            else:
                logits, _, past_key_values = self(next_token, past_key_values=past_key_values)

        if return_probs:
            return input_ids, sum(token_probs) / max(1, len(token_probs))
        return input_ids

    def save_pretrained(self, path: str):
        """Save model weights, config, and generation settings to directory."""
        os.makedirs(path, exist_ok=True)
        # Save architecture
        self.config.save(path)
        # Save weights (pytorch standard)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        # Save default generation settings
        from quantum_transformer.generation_config import GenerationConfig
        GenerationConfig().save(path)

    @classmethod
    def load_pretrained(cls, path: str, device: str = "cpu") -> "QuantumTransformerLM":
        """Load a pretrained model from directory (HF-style)."""
        config = QuantumTransformerConfig.load(path)
        model = cls(config)
        
        model_file = os.path.join(path, "model.pt")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Weight file not found in {path}")
            
        state_dict = torch.load(
            model_file,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    def get_architecture_info(self) -> dict:
        """Return a summary of the model architecture for the UI."""
        return {
            "num_qubits": self.config.num_qubits,
            "hidden_dim": self.config.hidden_dim,
            "num_heads": self.config.num_heads,
            "head_dim": self.config.head_dim,
            "num_layers": self.config.num_layers,
            "ffn_dim": self.config.ffn_dim,
            "max_context_length": self.config.max_context_length,
            "circuit_depth": self.config.circuit_depth,
            "state_space_size": self.config.state_space_size,
            "total_parameters": self.count_parameters(),
            "vocab_size": self.config.vocab_size,
        }
