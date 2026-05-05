
import torch
import os
import sys
import time

# Add parent dir to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_transformer.model import QuantumTransformerLM
from quantum_transformer.tokenizer import BPETokenizer

def test_model():
    model_path = "models/qt_4q_256ctx/"
    device = "cpu" # Test on CPU for stability
    
    print(f"⚛️ Loading model and tokenizer from {model_path}...")
    tokenizer = BPETokenizer()
    tokenizer.load_pretrained(model_path)
    model = QuantumTransformerLM.load_pretrained(model_path, device=device)
    
    questions = [
        "Who created you?",
        "What is a Qubit?",
        "What is the capital of France?",
        "Write a 4-line poem about a star."
    ]
    
    # We will test different "Repetition Penalties" to find the sweet spot
    # Current is -2.0 presence, -0.5 frequency
    
    print("\n" + "="*50)
    print("🚀 STARTING BENCHMARK (Temp=0.3, Precise Mode)")
    print("="*50)
    
    for q in questions:
        print(f"\n👤 USER: {q}")
        prompt = f"### USER:\n{q}\n\n### ASSISTANT:\n"
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        
        start_time = time.time()
        # Using the generate method we just updated
        output_ids, confidence = model.generate(
            input_ids, 
            max_new_tokens=100, 
            temperature=0.3,
            return_probs=True,
            stop_token_ids=[tokenizer.encode("\n\n")[0]]
        )
        duration = time.time() - start_time
        
        response = tokenizer.decode(output_ids[0].tolist())
        # Extract assistant part
        if "### ASSISTANT:\n" in response:
            response = response.split("### ASSISTANT:\n")[-1]
            
        print(f"⚛️ AI: {response}")
        print(f"📊 Confidence: {confidence:.2%} | Time: {duration:.2f}s")

if __name__ == "__main__":
    test_model()
