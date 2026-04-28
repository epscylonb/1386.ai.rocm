import torch
from src.model.transformer import Transformer
#from src.tokenizer import Tokenizer # Adjust based on your actual tokenizer path
import sentencepiece as spm

# 1. Setup
device = "cuda"
checkpoint_path = "checkpoints/1.1_step_11000.pt" # Or your latest
log = torch.load(checkpoint_path, map_location=device)

# 2. Load Model
# Ensure your config matches what you used for training
model = Transformer(log['config']) 
model.load_state_dict(log['model'])
model.to(device)
model.eval()

# 3. Simple Generate Function
def generate(prompt, max_new_tokens=50):
    # This assumes a basic tokenizer with an encode method
    idx = torch.tensor(spm.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    for _ in range(max_new_tokens):
        logits = model(idx[:, -1024:]) # Respect max seq len
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
        
    return spm.decode(idx[0].tolist())

# 4. Run Tests
#tokenizer = Tokenizer() # Initialize yours
print(generate("The capital of France is"))
print(generate("Deep learning is a subset of"))
