import torch
import sys
import os

def log(msg):
    print(f"[DEBUG] {msg}", flush=True)

log("--- ROCm Hardware Bootstrap Test ---")
log(f"Python version: {sys.version}")
log(f"PyTorch version: {torch.__version__}")
log(f"ROCm/HIP version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")

# 1. Basic Check
try:
    log("Checking torch.cuda.is_available()...")
    available = torch.cuda.is_available()
    log(f"is_available: {available}")
    
    if not available:
        log("ROCm not detected by PyTorch. Check your 'pip install' source.")
        sys.exit(1)

    log(f"Device Name: {torch.cuda.get_device_name(0)}")
    log(f"Device Count: {torch.cuda.device_count()}")
except Exception as e:
    log(f"CRASH during basic check: {e}")
    sys.exit(1)

# 2. Small Allocation (The "Handshake")
# This is usually where the -11 Segfault happens.
try:
    log("Attempting tiny tensor allocation on GPU...")
    # This triggers the first actual communication with the hardware
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    log("Allocation successful.")
    
    log("Testing simple math operation...")
    y = x * 2
    torch.cuda.synchronize()
    log(f"Math success. Result: {y.cpu()}")
except Exception as e:
    log(f"CRASH during allocation/math: {e}")
    sys.exit(1)

# 3. Large Allocation (The "Unified Memory" Test)
try:
    log("Attempting large allocation (1GB) to test Unified Memory mapping...")
    # 256M float32 elements = 1GB
    large_tensor = torch.zeros((1024, 1024, 256), device='cuda')
    torch.cuda.synchronize()
    log("Large allocation successful.")
except Exception as e:
    log(f"CRASH during large allocation: {e}")

log("--- Test Complete: No Segfault detected ---")