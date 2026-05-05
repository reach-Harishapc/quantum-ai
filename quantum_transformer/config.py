"""
Configuration for the Quantum Transformer architecture.

The key insight: each qubit maps to one attention head with dim_per_qubit
features. More qubits = larger hidden dimension = more model capacity.
This directly mirrors how adding qubits to a quantum computer exponentially
increases its computational state space.
"""

from dataclasses import dataclass, field, asdict
import json
import os


@dataclass
class QuantumTransformerConfig:
    """Configuration that fully defines a Quantum Transformer architecture."""
    # Metadata
    model_type: str = "quantum_transformer"
    version: str = "0.1.0"
    architectures: list = field(default_factory=lambda: ["QuantumTransformerLM"])

    # ── Quantum Parameters ──────────────────────────────────────────────
    num_qubits: int = 4
    """Number of simulated qubits. Each qubit becomes one attention head.
    Range: 1–32. Directly controls model size and capacity."""

    dim_per_qubit: int = 64
    """Feature dimensions allocated per qubit. Increased to 64 for 
    Massive Neuron Memory and high-level reasoning capacity."""

    circuit_depth: int = 3
    """Number of rotation + entanglement layers in each VQC.
    Deeper circuits = more expressive quantum transformations."""

    # ── Transformer Parameters ──────────────────────────────────────────
    max_context_length: int = 256
    """Maximum sequence length the model can process."""

    vocab_size: int = 8192
    """Vocabulary size. Defaults to 8192 for the new BPE tokenizer.
    If using legacy ByteTokenizer, set to 256."""

    dropout: float = 0.1
    """Dropout rate for regularization."""

    system_prompt: str = (
        "### Instruction:\nYou are Quantum AI, a world-class quantum-inspired language model "
        "created by Harisha P C, an Quantum AI researcher. "
        "Use your quantum-inspired logic to provide helpful and accurate professional responses.\n\n"
    )
    """Hardcoded system persona that defines the model's identity and behavior."""

    # ── Derived (computed) ──────────────────────────────────────────────
    @property
    def hidden_dim(self) -> int:
        """Total hidden dimension = num_qubits × dim_per_qubit."""
        return self.num_qubits * self.dim_per_qubit

    @property
    def num_heads(self) -> int:
        """Number of attention heads = num_qubits (1 head per qubit)."""
        return self.num_qubits

    @property
    def head_dim(self) -> int:
        """Dimension per attention head = dim_per_qubit."""
        return self.dim_per_qubit

    @property
    def num_layers(self) -> int:
        """Number of transformer blocks. Scales with qubits, capped at 8."""
        return min(self.num_qubits, 8)

    @property
    def ffn_dim(self) -> int:
        """Feed-forward intermediate dimension = 4 × hidden_dim.
        This represents the 'Permanent Memory' of the transformer."""
        return self.hidden_dim * 4

    @property
    def state_space_size(self) -> int:
        """Theoretical quantum state space = 2^num_qubits."""
        return 2 ** self.num_qubits

    @property
    def total_parameters_estimate(self) -> int:
        """Rough parameter count estimate."""
        d = self.hidden_dim
        L = self.num_layers
        V = self.vocab_size
        # Embedding + pos_emb + L*(attn + ffn + norms) + output
        emb = V * d + self.max_context_length * d
        attn_per_layer = 4 * d * d  # Q, K, V, out projections
        ffn_per_layer = 2 * d * self.ffn_dim  # up + down
        vqc_per_layer = self.circuit_depth * (d // 2 + self.num_qubits ** 2)  # rotations + entanglement
        norms = 2 * d * 2  # layernorm params
        output = d * V
        return emb + L * (attn_per_layer + ffn_per_layer + 2 * vqc_per_layer + norms) + output + d * 2

    def to_dict(self) -> dict:
        """Convert to dictionary, including all derived properties for full metadata."""
        data = asdict(self)
        data.update({
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "num_layers": self.num_layers,
            "ffn_dim": self.ffn_dim,
            "state_space_size": self.state_space_size,
            "total_parameters": self.total_parameters_estimate
        })
        return data

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "QuantumTransformerConfig":
        with open(os.path.join(path, "config.json"), "r") as f:
            data = json.load(f)
        
        # Filter out keys that are not part of the dataclass __init__
        # (derived properties like 'hidden_dim' that we save for metadata)
        import inspect
        sig = inspect.signature(cls)
        valid_params = [p.name for p in sig.parameters.values()]
        filtered_data = {k: v for k, v in data.items() if k in valid_params}
        
        return cls(**filtered_data)
