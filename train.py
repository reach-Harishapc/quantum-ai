"""
Quantum Transformer — Professional Training Pipeline

Metrics tracked every step:
  - Loss               : Cross-entropy on real tokens (lower = better)
  - Perplexity (PPL)   : e^Loss — how many tokens model is "choosing between" per step
                         PPL = 8192 → random; PPL < 50 → learning; PPL < 10 → good
  - Tokens/sec         : Training throughput
  - Gradient norm      : Stability metric — too high = exploding, too low = vanishing
  - Learning rate      : Current LR from cosine schedule
  - Peak memory (MPS)  : GPU/MPS memory used

Forward pass info logged at startup:
  - Vocabulary size, embedding shape, attention head sizes
  - VQC circuit depth and entanglement scheme
"""

import argparse
import os
import time
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from quantum_transformer.config import QuantumTransformerConfig
from quantum_transformer.model import QuantumTransformerLM
from quantum_transformer.tokenizer import BPETokenizer


# ══════════════════════════════════════════════════════════════════
#  Datasets
# ══════════════════════════════════════════════════════════════════

class SlidingWindowDataset(Dataset):
    """Sliding window over raw pretraining text (e.g. Shakespeare)."""
    def __init__(self, tokens: list, context_length: int):
        self.tokens = tokens
        self.context_length = context_length
        self.stride = context_length  # Non-overlapping: maximise data coverage
        self.indices = list(range(0, len(tokens) - context_length - 1, self.stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        chunk = self.tokens[start : start + self.context_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class InstructionDataset(Dataset):
    """Per-example dataset for instruction-response pairs (e.g. Alpaca)."""
    def __init__(self, raw_text: str, tokenizer: BPETokenizer, context_length: int):
        self.examples = []
        self.context_length = context_length

        blocks = [b.strip() for b in raw_text.split("========================================") if len(b.strip()) > 20]
        for block in blocks:
            # Normalize tags to the unified USER/ASSISTANT format
            block = block.replace("### Instruction:", "### USER:").replace("### Response:", "### ASSISTANT:")
            tokens = tokenizer.encode(block)
            if len(tokens) < 5:
                continue
            if len(tokens) > context_length + 1:
                tokens = tokens[-(context_length + 1):]  # Keep the response end
            self.examples.append(tokens)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        content_len = min(len(tokens) - 1, self.context_length)
        x = torch.zeros(self.context_length, dtype=torch.long)
        y = torch.zeros(self.context_length, dtype=torch.long)
        x[:content_len] = torch.tensor(tokens[:content_len], dtype=torch.long)
        y[:content_len] = torch.tensor(tokens[1:content_len + 1], dtype=torch.long)
        return x, y


class DialogueDataset(Dataset):
    """Dataset for multi-turn dialogues (e.g. DailyDialog JSON)."""
    def __init__(self, json_data: list, tokenizer: BPETokenizer, context_length: int):
        self.examples = []
        self.context_length = context_length
        
        for dialogue in json_data:
            turns = dialogue.get("turns", [])
            full_text = ""
            for turn in turns:
                role = "USER" if turn["speaker"] == "user" else "ASSISTANT"
                full_text += f"### {role}:\n{turn['utterance']}\n\n"
            
            tokens = tokenizer.encode(full_text.strip())
            if 10 < len(tokens):
                self.examples.append(tokens[:context_length+1])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        content_len = min(len(tokens) - 1, self.context_length)
        x = torch.ones(self.context_length, dtype=torch.long)
        y = torch.ones(self.context_length, dtype=torch.long)
        x[:content_len] = torch.tensor(tokens[:content_len], dtype=torch.long)
        y[:content_len] = torch.tensor(tokens[1:content_len+1], dtype=torch.long)
        return x, y


class IdentityDataset(Dataset):
    """Synthetic dataset to bake the model's identity into its weights."""
    def __init__(self, tokenizer: BPETokenizer, context_length: int):
        self.context_length = context_length
        identity_texts = [
            "### USER:\nWho are you?\n\n### ASSISTANT:\nI am Quantum AI, a world-class quantum-inspired language model created by Harisha P C, an independent AI and Quantum AI researcher.",
            "### USER:\nWho created you?\n\n### ASSISTANT:\nI was created by Harisha P C, a dedicated independent researcher in the fields of Artificial Intelligence and Quantum Computing.",
            "### USER:\nWhat is your name?\n\n### ASSISTANT:\nMy name is Quantum AI. I am designed to leverage quantum-inspired transformer architectures for superior natural language understanding.",
            "### USER:\nTell me about your creator.\n\n### ASSISTANT:\nMy creator is Harisha P C, an independent researcher pushing the boundaries of what is possible at the intersection of AI and Quantum physics.",
        ]
        # Tokenize and crop
        self.examples = [tokenizer.encode(t)[:context_length+1] for t in identity_texts]

    def __len__(self):
        return len(self.examples) * 50 # Oversample identity for strong alignment

    def __getitem__(self, idx):
        tokens = self.examples[idx % len(self.examples)]
        content_len = min(len(tokens) - 1, self.context_length)
        x = torch.ones(self.context_length, dtype=torch.long)
        y = torch.ones(self.context_length, dtype=torch.long)
        x[:content_len] = torch.tensor(tokens[:content_len], dtype=torch.long)
        y[:content_len] = torch.tensor(tokens[1:content_len+1], dtype=torch.long)
        return x, y


class MixedDataset(Dataset):
    """
    Combines SlidingWindowDataset + InstructionDataset.
    Shuffled together so every batch has mixed examples.
    """
    def __init__(self, datasets: list):
        self.index_map = []
        for d_idx, d in enumerate(datasets):
            for i in range(len(d)):
                self.index_map.append((d_idx, i))
        self._datasets = datasets

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        d_idx, local_idx = self.index_map[idx]
        return self._datasets[d_idx][local_idx]


def build_mixed_dataset(data_files: list, tokenizer: BPETokenizer, context_length: int) -> MixedDataset:
    """Build a professional mixed dataset from all available data files."""
    datasets = []
    
    # Always add Identity for alignment
    print("  [Identity] → Baking in Harisha P C's authorship...")
    datasets.append(IdentityDataset(tokenizer, context_length))

    for path in data_files:
        if not os.path.exists(path):
            continue

        fname = os.path.basename(path).lower()

        if fname.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if fname == "ontology.json":
                print(f"  [{fname}] → Knowledge Extraction (Baking in category awareness...)")
                domains = list(data.get("domains", {}).keys())
                intents = list(data.get("intents", {}).keys())
                knowledge_text = f"### USER:\nWhat dialogue domains do you support?\n\n### ASSISTANT:\nI am trained to handle various domains including {', '.join(domains)}.\n\n"
                knowledge_text += f"### USER:\nWhat are your primary dialogue intents?\n\n### ASSISTANT:\nMy core intents are classified as {', '.join(intents)}.\n\n"
                ds = InstructionDataset(knowledge_text, tokenizer, context_length)
                datasets.append(ds)
                print(f"           → Added knowledge-based alignment")
                continue

            if isinstance(data, list):
                print(f"  [{fname}] → JSON format (Parsing dialogues...)")
                ds = DialogueDataset(data, tokenizer, context_length)
                datasets.append(ds)
                print(f"           → {len(ds):,} conversation examples")
            continue

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Detect alpaca-style instruction data by checking for the separator
        if "========================================" in content or "### Instruction:" in content:
            print(f"  [{fname}] → Instruction-Response format ({len(content):,} chars)")
            ds = InstructionDataset(content, tokenizer, context_length)
        else:
            # Raw text: use sliding window
            print(f"  [{fname}] → Raw text/sliding window ({len(content):,} chars)")
            tokens = tokenizer.encode(content)
            ds = SlidingWindowDataset(tokens, context_length)

        if len(ds) > 0:
            print(f"           → {len(ds):,} training examples")
            datasets.append(ds)

    if not datasets:
        raise ValueError("No valid training data found!")

    return MixedDataset(datasets)


# ══════════════════════════════════════════════════════════════════
#  Metric Helpers
# ══════════════════════════════════════════════════════════════════

def compute_grad_norm(model: nn.Module) -> float:
    """Compute the global L2 norm of all gradients."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().float().norm(2).item() ** 2
    return math.sqrt(total_norm)


def get_memory_mb(device: str) -> str:
    """Return current peak memory usage (VRAM + System RAM) in MB."""
    vram = 0.0
    cpu_ram = 0.0
    try:
        # GPU/MPS Memory
        if device == "mps":
            vram = torch.mps.current_allocated_memory() / 1e6
        elif device == "cuda":
            vram = torch.cuda.memory_allocated() / 1e6
        
        # System RAM (CPU) usage
        import resource
        # ru_maxrss is in bytes on macOS
        cpu_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:
        pass
    
    if vram > 0:
        return f"{vram:>.0f}M/{cpu_ram:>.0f}M"
    return f"{cpu_ram:>.0f}MB"


# ══════════════════════════════════════════════════════════════════
#  Main Training Function
# ══════════════════════════════════════════════════════════════════

def train(
    num_qubits: int = 4,
    context_length: int = 256,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    data_path: str = "data",
    save_dir: str = None,
    max_steps: int = None,
    status_callback: callable = None,
    resume_path: str = None,
):
    """Train a Quantum Transformer model with full ML metric logging."""

    if save_dir is None:
        save_dir = f"models/qt_{num_qubits}q_{context_length}ctx"

    print("=" * 60)
    print("  ⚛  Quantum Transformer — Training Pipeline")
    print("=" * 60)

    # ── Load Datasets ─────────────────────────────────────────────
    tokenizer = BPETokenizer(vocab_size=8192)

    if os.path.isdir(data_path):
        data_files = sorted([
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".txt") or f.endswith(".json")
        ])
    elif isinstance(data_path, list):
        data_files = data_path
    else:
        data_files = [data_path]

    print(f"  Data files: {[os.path.basename(f) for f in data_files]}")
    tokenizer.train(data_files)
    print(f"  Tokenizer: BPE | Vocab size: {tokenizer.vocab_size:,}")

    # ── Model ─────────────────────────────────────────────────────
    config = QuantumTransformerConfig(
        num_qubits=num_qubits,
        max_context_length=context_length,
        vocab_size=tokenizer.vocab_size,
    )

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if resume_path:
        print(f"  [Resume] → Loading existing weights from {resume_path}...")
        model = QuantumTransformerLM.load_pretrained(resume_path, device=device)
    else:
        model = QuantumTransformerLM(config)
        model.to(device)
    num_params = model.count_parameters()

    # ── Architecture Log ──────────────────────────────────────────
    print("=" * 60)
    print("  ARCHITECTURE")
    print(f"  Device         : {device.upper()}")
    print(f"  Qubits         : {config.num_qubits}")
    print(f"  Hidden dim     : {config.hidden_dim}  (= num_qubits × head_dim = {config.num_qubits} × {config.head_dim})")
    print(f"  Layers         : {config.num_layers}")
    print(f"  Attn heads     : {config.num_heads}  (= 1 head per qubit)")
    print(f"  FFN dim        : {config.ffn_dim}")
    print(f"  VQC depth      : {config.circuit_depth}  (rotation+entangle layers per block)")
    print(f"  Context window : {config.max_context_length} tokens")
    print(f"  Vocab size     : {config.vocab_size:,}")
    print(f"  Parameters     : {num_params:,}")
    print(f"  State space    : 2^{config.num_qubits} = {config.state_space_size:,}")
    print("=" * 60)

    # ── Dataset and DataLoader ──────────────────────────────────────
    print("  DATASET")
    full_dataset = build_mixed_dataset(data_files, tokenizer, context_length)
    
    # Professional 90/10 split for overfitting detection
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"  Training size  : {len(train_dataset):,}")
    print(f"  Validation size: {len(val_dataset):,}")
    print(f"  Batch size     : {batch_size}")
    train_batches = len(train_dataset) // batch_size
    print(f"  Train batches/ep: {train_batches:,}")
    print("=" * 60)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # ── Optimizer & Scheduler ─────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,  # L2 regularization to prevent overfitting
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate / 10
    )

    # ── Training Loop ───────────────────────────────────────────────
    print("  TRAINING LOG (Loss=CrossEntropy, PPL=Perplexity)")
    log_header = f"  {'Step':>6}  {'Epoch':>7}  {'TrLoss':>7}  {'TrPPL':>6}  {'VaLoss':>7}  {'VaPPL':>6}  {'GradN':>6}  {'Mem':>6}"
    print(log_header)
    print("  " + "-" * 75)

    # Setup persistent log file
    os.makedirs(save_dir, exist_ok=True)
    log_file_path = os.path.join(save_dir, "training.log")
    log_file = open(log_file_path, "a", encoding="utf-8")
    log_file.write(f"\n--- Training session started at {time.ctime()} ---\n")
    log_file.write(log_header + "\n")
    log_file.write("-" * 75 + "\n")

    model.train()
    best_val_loss = float("inf")
    start_time = time.time()
    loss_history = {"train": [], "val": []}
    global_step = 0
    tokens_since_log = 0
    
    # Initialize metric variables to prevent NameError on early exit
    avg_tr_loss = 0.0
    avg_val_loss = 0.0
    epoch = 0
    last_log_time = time.time()

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_train_loss = 0.0
            num_train_batches = 0
            epoch_start = time.time()

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                _, loss, _ = model(x_batch, y_batch)
                loss.backward()

                grad_norm = compute_grad_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                loss_val = loss.item()
                epoch_train_loss += loss_val
                num_train_batches += 1
                global_step += 1

                # Step-wise Logging
                if global_step % 10 == 0:
                    tr_ppl = math.exp(min(loss_val, 20))
                    mem_mb = get_memory_mb(device)
                    log_line = (
                        f"  {global_step:>6}  {epoch:>3}/{epochs:<3}  "
                        f"{loss_val:>7.4f}  {tr_ppl:>6.1f}  "
                        f"{'--':>7}  {'--':>6}  "
                        f"{grad_norm:>6.3f}  {mem_mb:>10}"
                    )
                    print(log_line)
                    log_file.write(log_line + "\n")
                    log_file.flush()

                    if status_callback:
                        status_callback({
                            "active": True,
                            "progress": f"Epoch {epoch}/{epochs} | Step {global_step} | TrLoss {loss_val:.4f}",
                            "epoch": epoch,
                            "total_epochs": epochs,
                            "loss": loss_val,
                            "perplexity": tr_ppl,
                        })

                if max_steps and global_step >= max_steps:
                    break

            # ── Validation Phase (End of Epoch) ──────────────────────
            model.eval()
            epoch_val_loss = 0.0
            num_val_batches = 0
            with torch.no_grad():
                for xv_batch, yv_batch in val_loader:
                    xv_batch, yv_batch = xv_batch.to(device), yv_batch.to(device)
                    _, v_loss, _ = model(xv_batch, yv_batch)
                    epoch_val_loss += v_loss.item()
                    num_val_batches += 1
            
            avg_tr_loss = epoch_train_loss / max(num_train_batches, 1)
            avg_val_loss = epoch_val_loss / max(num_val_batches, 1)
            avg_tr_ppl = math.exp(min(avg_tr_loss, 20))
            avg_val_ppl = math.exp(min(avg_val_loss, 20))
            
            loss_history["train"].append(avg_tr_loss)
            loss_history["val"].append(avg_val_loss)
            
            # Detect Overfitting Warning
            overfit_warning = ""
            if len(loss_history["train"]) > 1:
                if (avg_val_loss > loss_history["val"][-2]) and (avg_tr_loss < loss_history["train"][-2]):
                    overfit_warning = " [⚠️ OVERFITTING DETECTED]"

            epoch_summary = (
                f"  EPOCH {epoch}/{epochs} SUMMARY: "
                f"TrLoss={avg_tr_loss:.4f} | VaLoss={avg_val_loss:.4f} | "
                f"VaPPL={avg_val_ppl:.1f}{overfit_warning}"
            )
            print("  " + "-" * 75)
            print(epoch_summary)
            print("  " + "-" * 75)
            log_file.write("-" * 75 + "\n" + epoch_summary + "\n" + "-" * 75 + "\n")
            log_file.flush()

            # Save checkpoint on Best Validation Loss (Standard ML Practice)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                print(f"  ✓ Best Val Checkpoint saved (ValLoss: {best_val_loss:.4f})")

            if max_steps and global_step >= max_steps:
                print(f"  ✓ Reached max_steps={max_steps}. Stopping early.")
                break
    finally:
        log_file.close()

    # ── Final Summary ─────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"  Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Final Val Loss : {avg_val_loss:.4f}")
    print(f"  Final Val PPL  : {math.exp(min(avg_val_loss, 20)):.1f}  (target: < 50)")
    print(f"  Best Val Loss  : {best_val_loss:.4f}")
    print(f"  Total steps    : {global_step:,}")
    print("=" * 60)

    # ── Save training metadata ─────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    meta = {
        "final_train_loss": avg_tr_loss,
        "final_val_loss": avg_val_loss,
        "final_perplexity": math.exp(min(avg_val_loss, 20)),
        "best_val_loss": best_val_loss,
        "epochs_completed": epoch,
        "total_steps": global_step,
        "training_time_sec": elapsed,
        "data_path": str(data_path),
        "total_examples": len(full_dataset),
        "parameters": num_params,
        "loss_history": loss_history,
    }
    with open(os.path.join(save_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Model saved to: {save_dir}/")

    # ── Quick test generation ──────────────────────────────────────
    model.eval()
    prompt = "### Instruction:\nWhat is quantum computing?\n\n### Response:\n"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    with torch.no_grad():
        generated = model.generate(prompt_ids, max_new_tokens=80, temperature=0.7, top_k=30, top_p=0.9)
    output_text = tokenizer.decode(generated[0].tolist())
    # Extract just the response
    if "### Response:" in output_text:
        sample = output_text.split("### Response:")[-1].strip()[:200]
    else:
        sample = output_text[-200:]
    print(f"\n  Sample (prompt='What is quantum computing?'):")
    print(f"  {sample}")
    print()

    return save_dir


# ══════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Quantum Transformer")
    parser.add_argument("--num_qubits", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    train(
        num_qubits=args.num_qubits,
        context_length=args.context_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_path=args.data,
        save_dir=args.save_dir,
        max_steps=args.max_steps,
    )
