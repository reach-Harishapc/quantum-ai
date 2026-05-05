# Quantum Transformer Architecture

This document describes the underlying mechanisms and architecture of the **Quantum-Inspired Transformer**. The project builds a novel language modeling architecture in PyTorch by injecting constraints and operations inspired by quantum computing into standard transformer blocks.

## Core Concepts

The architecture differs from a standard classical Transformer (like GPT-2) by replacing simple linear attention projections and feed-forward layers with **Variational Quantum Circuits (VQC)**. The model is fully differentiable and trained using standard backpropagation, but its internal representations are forced through "quantum gates."

### 1. Quantum State Representation
In this model, the hidden dimension is divided into smaller chunks called **qubits**.
*   **1 Qubit = 1 Attention Head**
*   **State Space per Qubit**: `dim_per_qubit` (e.g., 16 or 32 features).
*   **Total Hidden Dimension**: `num_qubits * dim_per_qubit`.

Instead of operating on the entire vector space arbitrarily, the model treats the space as a collection of separable qubits that only interact under specific controlled entanglement operations.

---

### 2. Variational Quantum Circuits (VQCs)
The standard Linear layers are replaced by simulated quantum circuits inside both the **Attention Mechanism** and the **Feed Forward Network (FFN)**. 

A VQC in this architecture consists of stacked layers of alternating operations:
1.  **Quantum Rotations ($R$ gates)**
2.  **Quantum Entanglement**

`Circuit Depth` determines how many times these two operations are alternately applied to the input state.

#### A. Quantum Rotation (Givens Rotations)
In quantum computing, single-qubit gates (like $R_y$, $R_z$, $R_x$) rotate the state vector of a single qubit. To simulate this expressivity in real-valued PyTorch tensors without complex numbers, we use **Parameterized Givens Rotations**. 

*   A Givens rotation is a 2D rotation matrix that mixes pairs of features within a single qubit's state space by an angle $\theta$.
*   The model learns exactly `dim_per_qubit // 2` rotation angles per qubit.
*   This mimics applying an $R$ gate to a superposition state. It preserves the vector's norm and serves as a highly structured, orthogonal linear transformation.

#### B. Quantum Entanglement 
Without entanglement, quantum computers (and this model) would just operate on independent feature spaces (qubits). To create complex, non-linear relationships, the qubits must interact. 

*   Instead of classic CNOT gates applied between adjacent qubits, the model uses a **Learned Multi-stage Mixing Matrix**.
*   This acts as a self-attention mechanism *across the qubits themselves*. Each qubit's state space is mapped to every other qubit using a learned $(Q \times Q)$ weighted interaction matrix (passed through a Softmax).
*   This mimics complex multi-qubit entanglement interactions over higher-dimensional feature spaces, allowing "teleportation" of context between the model's simulated quantum states.

---

### 3. The Quantum Attention Mechanism
In a classic multi-head attention mechanism:
*   $Q = X W_q$
*   $K = X W_k$
*   $V = X W_v$

In the **Quantum Transformer**:
*   The Query ($Q$) and Key ($K$) projections are not simple linear transforms. Instead, the input embeddings $X$ are passed through a Variational Quantum Circuit (VQC) with a defined `circuit_depth`. 
*   Because the VQC uses orthogonal rotations and entanglement, it creates a much more constrained and theoretically robust representation space.
*   The Attention score is determined by how aligned the "measured states" of the Query VQC and Key VQC are. 

### 4. Advanced Optimizations
#### KV-Caching (Key-Value Caching)
To achieve world-class inference speeds, the model implements KV-caching. During the autoregressive generation loop, the model avoids redundant Variational Quantum Circuit (VQC) computations by storing the Key and Value states of previous tokens. This reduces the complexity of each new token generation from O(N²) to O(N).

#### Instruction-Aware Response Pattern
The model is specifically conditioned for instruction-following tasks. It utilizes a strict template structure:
- `### Instruction:` (User Intent)
- `### Response:` (Quantum-Optimized Output)
- `========================================` (Sequence Boundary)

#### Geometric Similarity Search
Instead of traditional dot-product attention, the model uses normalized L2-similarity (Cosine Similarity) with a learned temperature factor. This forces the attention mechanism to focus on the geometric overlap of state vectors in the simulated Hilbert space, providing sharper and more precise 'probability-based' answers.

### 5. Why "Quantum Inspired"?
Current quantum computers don't have the coherence times or qubit counts to run a full LLM. This model proves that the *mathematical structure* of quantum computing (unitary rotations, localized state spaces, controlled entanglement mixing matrices) serves as a powerful **Inductive Bias** for neural networks. It requires fewer parameters to learn complex rotational geometries than a massive standard linear layer.

---
*Developed for the world: High-Fidelity Quantum Transformer Module v1.0*
