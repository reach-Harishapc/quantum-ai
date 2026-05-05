"""
Quantum-inspired neural network layers.

These layers implement the core quantum operations as differentiable
PyTorch modules:

# 1. QuantumRotation — Parameterized Givens rotations acting on pairs of
#    features. This is the classical analog of single-qubit rotation gates
#    (RY).
#
# 2. QuantumPhaseShift — Adds learned phase shifts (RZ/P) to features, 
#    mimicking quantum phase accumulation.
#
# 3. QuantumEntanglement — Learned multi-qubit mixing using a full 
#    correlation matrix.
#
# 4. VariationalQuantumCircuit (VQC) — Alternating layers of rotation,
#    phase, and entanglement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ═══════════════════════════════════════════════════════════════════════
# Quantum Gate Layers (Core Components)
# ═══════════════════════════════════════════════════════════════════════

def orthogonal_init_(tensor, gain=1.0):
    """Module-grade orthogonal initialization for near-unitary transforms."""
    with torch.no_grad():
        if tensor.ndimension() < 2:
            return tensor.fill_(0.0)
        nn.init.orthogonal_(tensor, gain=gain)
    return tensor

class QuantumRotation(nn.Module):
    """
    Parameterized Givens rotation gate.
    Mathematical equivalent of RY(θ) gates in a Hilbert space.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.theta = nn.Parameter(torch.zeros(dim // 2))
        self.gain = nn.Parameter(torch.ones(1) * 0.1) # Learned magnitude control

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.view(-1, self.dim // 2, 2)
        
        c = torch.cos(self.theta)
        s = torch.sin(self.theta)

        # Apply Unitary Rotation: [cos -sin; sin cos]
        y_even = c * x[..., 0] - s * x[..., 1]
        y_odd = s * x[..., 0] + c * x[..., 1]
        
        y = torch.stack([y_even, y_odd], dim=-1)
        # Residual with learned gain for numerical stability
        return (x + self.gain * y).view(orig_shape)

class QuantumPhaseShift(nn.Module):
    """Simulates parameterized phase accumulation (RZ gates)."""
    def __init__(self, dim: int):
        super().__init__()
        self.phi = nn.Parameter(torch.zeros(dim))
        self.gain = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Complex rotation simulation in real-space
        phase_res = x * torch.cos(self.phi) - x * torch.sin(self.phi)
        return x + self.gain * phase_res

class QuantumEntanglement(nn.Module):
    """
    Quantum entanglement layer using high-rank correlation matrices.
    Provides non-local interaction between simulated qubit subspaces.
    """
    def __init__(self, num_qubits: int, dim_per_qubit: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.dim_per_qubit = dim_per_qubit

        if num_qubits > 1:
            # Multi-layer mixing: (Q, Q) matrix for relationship mapping
            self.mixing = nn.Parameter(torch.randn(num_qubits, num_qubits) * 0.02)
            self.register_buffer("identity_mix", torch.eye(num_qubits) * 2.0)
        else:
            self.register_buffer("mixing", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_qubits <= 1: return x

        orig_shape = x.shape
        x_reshaped = x.view(*orig_shape[:-1], self.num_qubits, self.dim_per_qubit)

        # Normalized mixing matrix (simulating multi-qubit gates)
        weights = F.softmax(self.mixing + self.identity_mix, dim=-1)
        out = torch.matmul(weights, x_reshaped)
        
        return out.reshape(orig_shape)

class VariationalQuantumCircuit(nn.Module):
    """
    Module-grade Variational Quantum Circuit (VQC) ansatz.
    Consists of alternating rotation, phase, and entanglement layers.
    """
    def __init__(self, hidden_dim: int, num_qubits: int,
                 dim_per_qubit: int, depth: int = 3):
        super().__init__()
        self.depth = depth

        self.rotations = nn.ModuleList([QuantumRotation(hidden_dim) for _ in range(depth)])
        self.phase_shifts = nn.ModuleList([QuantumPhaseShift(hidden_dim) for _ in range(depth)])
        self.entanglements = nn.ModuleList([
            QuantumEntanglement(num_qubits, dim_per_qubit) for _ in range(depth)
        ])
        
        # Intermediate projections to bridge quantum states
        self.projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)
        ])
        
        # Initialize projections as near-identity orthogonal transforms
        for proj in self.projections:
            orthogonal_init_(proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for rot, phase, ent, proj in zip(self.rotations, self.phase_shifts, self.entanglements, self.projections):
            x = rot(x)
            x = phase(x)
            x = ent(x)
            x = F.gelu(proj(x)) # Measurement-inspired nonlinearity
        return x


# ═══════════════════════════════════════════════════════════════════════
# Quantum Transformer Building Blocks
# ═══════════════════════════════════════════════════════════════════════

class QuantumMultiHeadAttention(nn.Module):
    """
    High-Fidelity Quantum Multi-Head Self-Attention.
    Uses Variational Quantum Circuits (VQC) to transform Queries and Keys
    into a high-dimensional Hilbert space before computing similarity.
    """
    def __init__(self, hidden_dim: int, num_heads: int,
                 head_dim: int, num_qubits: int,
                 circuit_depth: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim

        # High-precision projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Unitary initialization for stability
        orthogonal_init_(self.q_proj.weight)
        orthogonal_init_(self.k_proj.weight)
        orthogonal_init_(self.v_proj.weight)
        orthogonal_init_(self.out_proj.weight)

        # Quantum state transformation circuits
        self.q_circuit = VariationalQuantumCircuit(hidden_dim, num_qubits, head_dim, circuit_depth)
        self.k_circuit = VariationalQuantumCircuit(hidden_dim, num_qubits, head_dim, circuit_depth)

        self.attn_dropout = nn.Dropout(dropout)
        # Softmax Sharpness control (Init to 1/sqrt(d_k) for stability)
        self.tau = nn.Parameter(torch.ones(num_heads, 1, 1) * (1.0 / math.sqrt(self.head_dim)))

    def forward(self, x: torch.Tensor, 
                mask: torch.Tensor = None,
                past_key_value: tuple = None) -> tuple:
        B, S, D = x.shape

        # Transform to Quantum State Space
        Q = self.q_circuit(self.q_proj(x))
        K = self.k_circuit(self.k_proj(x))
        V = self.v_proj(x)

        # Reshape for multi-head processing: (B, H, S, D_h)
        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # ── KV Cache Management ──
        if past_key_value is not None:
            prev_K, prev_V = past_key_value
            K = torch.cat([prev_K, K], dim=2)
            V = torch.cat([prev_V, V], dim=2)
        
        current_kv = (K, V)

        # ── Quantum Similarity Search ──
        # Normalize to unit sphere for geometric consistency (Hilbert Space Overlap)
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)

        # Compute Similarity with Learned Temperature scaling
        scores = torch.matmul(Q, K.transpose(-2, -1)) * torch.exp(self.tau)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax yields the actual 'Probability' of correlation
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Final projection and head recombination
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out), current_kv



class QuantumFeedForward(nn.Module):
    """
    Quantum-enhanced Feed-Forward Network.

    Standard FFN with a QuantumRotation gate applied after the up-projection.
    The rotation provides a structured, quantum-inspired nonlinearity
    before the GELU activation.

    Structure: Linear↑ → QuantumRotation → GELU → Dropout → Linear↓
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.up = nn.Linear(hidden_dim, ffn_dim)
        self.rotation = QuantumRotation(ffn_dim)
        self.phase = QuantumPhaseShift(ffn_dim)
        self.down = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.rotation(x)
        x = self.phase(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.down(x)
        return x


class QuantumTransformerBlock(nn.Module):
    """
    Module-grade Quantum Transformer block with KV Caching support.
    """
    def __init__(self, hidden_dim: int, num_heads: int,
                 head_dim: int, num_qubits: int,
                 ffn_dim: int, circuit_depth: int,
                 dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = QuantumMultiHeadAttention(
            hidden_dim, num_heads, head_dim, num_qubits,
            circuit_depth, dropout
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = QuantumFeedForward(hidden_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
        # World-Class Learned Residual Scaling for stability
        self.res_alpha1 = nn.Parameter(torch.ones(1) * 0.1) 
        self.res_alpha2 = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None,
                past_key_value: tuple = None) -> tuple:
        # Pre-Norm + Gated Residual Attention
        norm_x = self.ln1(x)
        attn_out, kv = self.attn(norm_x, mask=mask, past_key_value=past_key_value)
        x = x + self.res_alpha1 * self.dropout(attn_out)
        
        # Pre-Norm + Gated Residual FFN
        x = x + self.res_alpha2 * self.dropout(self.ffn(self.ln2(x)))
        return x, kv
