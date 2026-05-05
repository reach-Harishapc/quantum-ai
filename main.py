import os
import glob
import json
import threading
import time
import torch
import logging
import sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, Any, Optional

from quantum_transformer.config import QuantumTransformerConfig
from quantum_transformer.model import QuantumTransformerLM
from quantum_transformer.tokenizer import BPETokenizer

# Setup logging to a file so we can see what's happening inside the bundle
logging.basicConfig(
    filename='debug.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Path Resolution for Desktop / PyInstaller
# ═══════════════════════════════════════════════════════════════════════

def get_base_path():
    """Get the base path of the application (where the executable is)"""
    if getattr(sys, 'frozen', False):
        # If bundled, the executable is here
        return os.path.dirname(sys.executable)
    return os.path.abspath(".")

def resource_path(relative_path):
    """Get absolute path to bundled resources (templates/static)"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Set the working directory to where the app is located
# This ensures "models" and "data" are looked for next to the app icon
BASE_DIR = get_base_path()
os.chdir(BASE_DIR)

app = FastAPI(title="Quantum AI Pretrained Model Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use resource_path for static and templates (bundled inside the app)
app.mount("/static", StaticFiles(directory=resource_path("static")), name="static")

# Fix for Python 3.14 / Jinja2 cache bug: Disable cache explicitly
from jinja2 import Environment, FileSystemLoader
jinja_env = Environment(loader=FileSystemLoader(resource_path("templates")), cache_size=0)
templates = Jinja2Templates(env=jinja_env)

# ═══════════════════════════════════════════════════════════════════════
# Global State
# ═══════════════════════════════════════════════════════════════════════

tokenizer = BPETokenizer()
current_model: Optional[QuantumTransformerLM] = None
current_model_path: Optional[str] = None
chat_history = []
training_status = {"active": False, "progress": "", "epoch": 0, "total_epochs": 0, "loss": 0.0}

# Models and data stay next to the application
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR, exist_ok=True)
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Request / Response Models
# ═══════════════════════════════════════════════════════════════════════

class ModelLoadRequest(BaseModel):
    num_qubits: int
    max_context_length: int

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9

class TrainRequest(BaseModel):
    num_qubits: int
    max_context_length: int
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 6e-4
    max_steps: Optional[int] = None
    training_mode: str = "new"


# ═══════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════

def get_model_dir(num_qubits: int, ctx_len: int) -> str:
    return os.path.join(MODELS_DIR, f"qt_{num_qubits}q_{ctx_len}ctx")


def list_available_models() -> list:
    """Scan the models directory for available pretrained models."""
    models = []
    if not os.path.exists(MODELS_DIR):
        return models

    for model_dir in sorted(glob.glob(os.path.join(MODELS_DIR, "qt_*"))):
        config_path = os.path.join(model_dir, "config.json")
        model_path = os.path.join(model_dir, "model.pt")
        if os.path.exists(config_path) and os.path.exists(model_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            meta_path = os.path.join(model_dir, "training_meta.json")
            # Auto-scan data directory for all relevant files
            data_files = []
            if os.path.exists("data"):
                data_files = [
                    os.path.join("data", f) 
                    for f in os.listdir("data") 
                    if f.endswith(".txt") or f.endswith(".json")
                ]
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            models.append({
                "path": model_dir,
                "num_qubits": config["num_qubits"],
                "max_context_length": config["max_context_length"],
                "parameters": meta.get("parameters", 0),
                "final_loss": meta.get("final_loss", 0),
                "training_time": meta.get("training_time", 0),
            })
    return models


def load_model(model_dir: str):
    """Load a pretrained model and its tokenizer into memory."""
    global current_model, current_model_path, tokenizer
    current_model = QuantumTransformerLM.load_pretrained(model_dir)
    current_model_path = model_dir
    # Load the specific BPE tokenizer for this model
    tokenizer = BPETokenizer()
    tokenizer.load_pretrained(model_dir)


def run_training_background(req: TrainRequest):
    """Run training in a background thread."""
    global training_status
    from train import train as train_model

    def update_status(status_dict):
        global training_status
        training_status.update(status_dict)

    training_status = {
        "active": True,
        "progress": "Initializing training...",
        "epoch": 0,
        "total_epochs": req.epochs,
        "loss": 0.0,
    }

    try:
        # Determine if we should resume
        resume_path = None
        if req.training_mode in ["continue", "finetune"]:
            potential_path = os.path.join(MODELS_DIR, f"qt_{req.num_qubits}q_{req.max_context_length}ctx")
            if os.path.exists(os.path.join(potential_path, "model.pt")):
                resume_path = potential_path
                print(f"⚛️ Resuming from: {resume_path} (Mode: {req.training_mode})")

        save_dir = train_model(
            num_qubits=req.num_qubits,
            context_length=req.max_context_length,
            epochs=req.epochs,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            data_path="data",
            max_steps=req.max_steps,
            resume_path=resume_path, # Passed as resume_path now
            status_callback=update_status
        )
        # Auto-load the newly trained model
        load_model(save_dir)
        training_status["active"] = False
        training_status["progress"] = "Training complete! Model loaded."
    except Exception as e:
        import traceback
        print(f"Training Error: {e}")
        traceback.print_exc()
        training_status["active"] = False
        training_status["progress"] = f"Training failed: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════

@app.get("/")
async def serve_ui(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.exception("Error serving UI:")
        return {"error": "Failed to load UI. Check debug.log for details.", "details": str(e)}


@app.get("/api/models")
async def api_list_models():
    """List all available pretrained models."""
    return {"models": list_available_models()}


@app.get("/api/config")
async def api_get_config():
    """Get current loaded model config."""
    if current_model is None:
        return {
            "loaded": False,
            "message": "No model loaded. Train or load a model first.",
            "available_models": list_available_models(),
        }

    info = current_model.get_architecture_info()
    info["loaded"] = True
    info["model_path"] = current_model_path

    # Build circuit visualization data
    info["circuit"] = {
        "qubits": info["num_qubits"],
        "wires": [{"id": i, "label": f"|q_{i}⟩"} for i in range(info["num_qubits"])],
        "circuit_depth": info["circuit_depth"],
        "state_space": info["state_space_size"],
        "gates_per_layer": info["num_qubits"] * 2,
    }
    return info


@app.post("/api/config")
async def api_load_model(req: ModelLoadRequest):
    """Load a pretrained model with specified config."""
    if req.num_qubits < 1 or req.num_qubits > 32:
        raise HTTPException(status_code=400, detail="Qubits must be between 1 and 32")

    model_dir = get_model_dir(req.num_qubits, req.max_context_length)

    if not os.path.exists(os.path.join(model_dir, "model.pt")):
        return {
            "loaded": False,
            "message": f"No pretrained model found for {req.num_qubits}q / {req.max_context_length}ctx. Train one first.",
            "available_models": list_available_models(),
        }

    try:
        load_model(model_dir)
        info = current_model.get_architecture_info()
        info["loaded"] = True
        info["model_path"] = current_model_path
        info["circuit"] = {
            "qubits": info["num_qubits"],
            "wires": [{"id": i, "label": f"|q_{i}⟩"} for i in range(info["num_qubits"])],
            "circuit_depth": info["circuit_depth"],
            "state_space": info["state_space_size"],
            "gates_per_layer": info["num_qubits"] * 2,
        }
        info["message"] = f"Model loaded: {req.num_qubits}-qubit Quantum Transformer"
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """Generate text using high-fidelity quantum transformer inference."""
    global chat_history
    if current_model is None:
        raise HTTPException(status_code=400, detail="No model loaded.")

    try:
        device = next(current_model.parameters()).device
        
        # 1. Build the History-aware prompt
        chat_history.append({"role": "user", "content": req.prompt})
        
        # Keep only last 10 turns to stay within context window
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
            
        # 1. Build conversation context with System Prompt from Model Config
        system_prompt = getattr(current_model.config, "system_prompt", "### USER:\nYou are Quantum AI, a helpful AI assistant.\n\n### ASSISTANT:\nHow can I help you today?\n\n")
        
        full_conversation = system_prompt
        for msg in chat_history:
            role = "USER" if msg["role"] == "user" else "ASSISTANT"
            full_conversation += f"### {role}:\n{msg['content']}\n\n"
        
        full_conversation += "### ASSISTANT:\n"
        
        prompt_ids_list = tokenizer.encode(full_conversation)
        
        # 2. Strict Context Truncation
        # Leave at least 64 tokens for the response
        max_prompt_len = current_model.config.max_context_length - 64
        if len(prompt_ids_list) > max_prompt_len:
            prompt_ids_list = prompt_ids_list[-max_prompt_len:]
            
        prompt_ids = torch.tensor([prompt_ids_list], dtype=torch.long).to(device)

        # 3. Professional Inference with KV Cache and Stop Tokens
        # Derive stop token IDs from the instruction separator
        stop_strings = ["### Instruction:", "========================================"]
        stop_token_ids = []
        for s in stop_strings:
            ids = tokenizer.encode(s)
            if ids:
                stop_token_ids.append(ids[0])  # Stop on first token of separator

        start_time = time.time()
        with torch.no_grad():
            generated, mean_prob = current_model.generate(
                prompt_ids,
                max_new_tokens=min(req.max_tokens, 256),
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                repetition_penalty=1.1,
                stop_token_ids=stop_token_ids,
                return_probs=True
            )
        generation_time = time.time() - start_time

        full_text = tokenizer.decode(generated[0].tolist())
        
        # 3. Precise response extraction logic
        # Split on the LAST occurrence of ASSISTANT: to get the model's new reply
        parts = full_text.split("### ASSISTANT:")
        response_part = parts[-1].strip() if len(parts) > 1 else full_text
        
        # Stop at common separators
        response_text = response_part
        for sep in ["### USER:", "========================================", "### ASSISTANT:"]:
            if sep in response_text:
                # If the model starts repeating the tags, cut it off
                response_text = response_text.split(sep)[0]
        
        response_text = response_text.strip()
        
        # 4. Save to history
        chat_history.append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "prompt": req.prompt,
            "metadata": {
                "model": f"{current_model.config.num_qubits}-Qubit Quantum Transformer",
                "parameters": current_model.count_parameters(),
                "generation_time": f"{generation_time:.2f}s",
                "confidence": f"{mean_prob:.2%}",
                "sampling": {
                    "temp": req.temperature,
                    "top_p": req.top_p,
                    "top_k": req.top_k
                }
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/api/train")
async def api_train(req: TrainRequest):
    """Start training a new model (runs in background thread)."""
    if training_status["active"]:
        raise HTTPException(status_code=409, detail="Training already in progress.")

    if req.num_qubits < 1 or req.num_qubits > 32:
        raise HTTPException(status_code=400, detail="Qubits must be between 1 and 32")

    thread = threading.Thread(target=run_training_background, args=(req,))
    thread.daemon = True
    thread.start()

    return {"message": f"Training started for {req.num_qubits}-qubit model.", "status": "training"}


@app.get("/api/train/status")
async def api_train_status():
    """Check training progress."""
    return training_status


# ═══════════════════════════════════════════════════════════════════════
# Startup: auto-load first available model
# ═══════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    models = list_available_models()
    if models:
        try:
            # Sort models to find the 'Best' one (most qubits, then most context)
            best_model = sorted(models, key=lambda x: (x['num_qubits'], x['max_context_length']), reverse=True)[0]
            load_model(best_model["path"])
            print(f"Auto-loaded best model: {best_model['path']} ({best_model['num_qubits']}q / {best_model['max_context_length']}ctx)")
        except Exception as e:
            print(f"Failed to auto-load model: {e}")
    else:
        print("No pretrained models found. Train one first.")


if __name__ == "__main__":
    import uvicorn
    import webview
    import threading
    
    # Function to start the FastAPI server in a background thread
    def start_fastapi():
        logger.info("🚀 Starting FastAPI Server for Desktop Window...")
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

    # Start FastAPI in a separate thread
    server_thread = threading.Thread(target=start_fastapi)
    server_thread.daemon = True
    server_thread.start()
    
    # Wait a moment for server to initialize
    time.sleep(1.5)
    
    # Create the standalone desktop window
    logger.info("🖥️ Launching Native Desktop Window...")
    webview.create_window(
        'Quantum AI - Desktop', 
        'http://127.0.0.1:8000',
        width=1200,
        height=800,
        min_size=(1000, 700),
        text_select=True,
        background_color='#000000'
    )
    
    # Start the webview loop (this blocks until the window is closed)
    webview.start()
