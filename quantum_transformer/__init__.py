"""
Quantum Transformer — A quantum-inspired transformer architecture for
building pretrained language models.

This module implements parameterized quantum gates (Givens rotations),
quantum entanglement layers, and variational quantum circuits as
differentiable PyTorch modules. These are composed into a full
autoregressive transformer that can be trained on text data.

Architecture mapping:
  - num_qubits  → number of attention heads & qubit subspaces
  - dim_per_qubit (16) → dimension per attention head
  - hidden_dim = num_qubits × dim_per_qubit
  - More qubits = larger model = more capacity

Core quantum components:
  - QuantumRotation: Parameterized Givens rotations (single-qubit gates)
  - QuantumEntanglement: Learned mixing across qubit subspaces (CNOT-inspired)
  - VariationalQuantumCircuit: Stacked rotation + entanglement layers
"""

from quantum_transformer.config import QuantumTransformerConfig
from quantum_transformer.model import QuantumTransformerLM
from quantum_transformer.tokenizer import ByteTokenizer

__all__ = ["QuantumTransformerConfig", "QuantumTransformerLM", "ByteTokenizer"]
