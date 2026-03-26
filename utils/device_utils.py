import torch
import os
import subprocess

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

def check_gpu_compatibility():
    """Return (ok, message) after a minimal CUDA tensor smoke test."""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    try:
        test_tensor = torch.tensor([1.0]).cuda()
        test_result = test_tensor + 1
        return True, "GPU OK"
    except RuntimeError as e:
        if "no kernel image is available" in str(e) or "not compatible" in str(e):
            return False, f"GPU not compatible: {e}"
        else:
            return False, f"GPU test failed: {e}"
    except Exception as e:
        return False, f"GPU check error: {e}"

def check_cuda_details():
    """Print CUDA/PyTorch diagnostics; return True if CUDA is usable."""
    print("=== CUDA check ===")
    
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA toolkit (PyTorch): {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        is_compatible, message = check_gpu_compatibility()
        if is_compatible:
            print(f"GPU compatibility: {message}")
        else:
            print(f"GPU compatibility: {message}")
            print("Suggestions:")
            print("  1. Install PyTorch nightly or a build that supports your GPU")
            print("  2. Run on CPU")
            return False
    else:
        print("CUDA not available. Common causes:")
        print("1. CPU-only PyTorch build")
        print("2. Missing or wrong NVIDIA driver")
        print("3. CUDA version mismatch")
        
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("\nNVIDIA driver detected:")
                print(result.stdout)
                print("Suggestion: reinstall PyTorch with CUDA support for your driver")
            else:
                print("nvidia-smi failed; install NVIDIA drivers if you need GPU")
        except FileNotFoundError:
            print("nvidia-smi not found; install NVIDIA drivers if you need GPU")
        return False
    
    print("==================")
    return True

def get_device():
    """Pick device: cuda if compatible, else mps (macOS), else cpu."""
    if not hasattr(get_device, '_detailed_check_done'):
        cuda_compatible = check_cuda_details()
        get_device._detailed_check_done = True
        
        if not cuda_compatible:
            print("GPU not usable; using CPU")
            device = 'cpu'
            print(f"Device: {device}")
            return device
    
    if torch.cuda.is_available():
        is_compatible, message = check_gpu_compatibility()
        if is_compatible:
            device = 'cuda'
            print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
        else:
            print(f"GPU issue: {message}")
            print("Falling back to CPU")
            device = 'cpu'
            print(f"Device: {device}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print(f"Device: {device}")
    else:
        device = 'cpu'
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print("Warning: CUDA reported but GPU not compatible; using CPU")
    
    return device

def get_cuda_installation_guide():
    """Print hints for very new GPUs (e.g. RTX 50 series) and nightly wheels."""
    guide = """
=== GPU / PyTorch install hints ===

If your GPU is newer than your PyTorch build (e.g. some RTX 50 series cards):

1) Try PyTorch nightly (example CUDA 12.4 wheels; check pytorch.org for current index):
   pip uninstall torch torchvision torchaudio -y
   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

2) Or run on CPU (this app will fall back automatically).

3) See https://pytorch.org/get-started/locally/ for the latest supported matrix.

=================================
"""
    print(guide)
    return guide
