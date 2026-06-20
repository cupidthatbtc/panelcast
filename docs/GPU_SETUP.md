# GPU Setup Guide for WSL2

This guide covers setting up GPU acceleration for JAX/NumPyro MCMC sampling on Windows using WSL2.

## Prerequisites

- **Windows 11** (or Windows 10 version 21H2+)
- **NVIDIA GPU** with CUDA support (tested on RTX 5090)
- **WSL2** enabled with Ubuntu distribution
- **Python 3.11+** (required for JAX >= 0.4.38)

## Windows Setup

### Install NVIDIA Driver

1. Download the driver from [NVIDIA Driver Downloads](https://www.nvidia.com/download/index.aspx)
   - Select your GPU model (e.g., GeForce RTX 5090)
   - Select Windows 11 64-bit
   - Either Game Ready or Studio driver works

2. Install the driver and **reboot Windows**

3. The Windows driver automatically provides `libcuda.so` to WSL2 via stub library passthrough

**Required driver versions:**
- RTX 5090 (Blackwell): Driver 570+
- RTX 40 series: Driver 525+
- RTX 30 series: Driver 470+

## WSL2 Setup

### Step 1: Verify GPU Passthrough

Open WSL2 terminal and run:

```bash
nvidia-smi
```

Expected output shows GPU name, driver version, and memory. If this fails:
- Verify Windows driver is installed correctly
- Check WSL2 version: `wsl --version` (kernel should be 5.10.16.3+)
- Update WSL: `wsl --update`

### Step 2: Install CUDA Toolkit

**CRITICAL:** Install ONLY the CUDA toolkit, NOT drivers. Installing Linux GPU drivers in WSL2 will break GPU access.

```bash
# Add CUDA repository (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install toolkit ONLY - do NOT install 'cuda' or 'cuda-drivers' packages
sudo apt-get install -y cuda-toolkit-12-8
```

For other versions, replace `cuda-toolkit-12-8` with appropriate version (e.g., `cuda-toolkit-12-6`).

## Python Environment Setup

### Install JAX with CUDA Support

```bash
# Upgrade pip first
pip install --upgrade pip

# Install JAX with CUDA 12 support
pip install --upgrade "jax[cuda12]"
```

This installs JAX with bundled CUDA libraries, avoiding version conflicts.

### Verify JAX Version

```bash
python -c "import jax; print(jax.__version__)"
```

Ensure version is **0.4.38 or higher** for RTX 5090 (Blackwell sm_120) support.

## Verification

Run the verification script from the project root:

```bash
python scripts/verify_gpu.py
```

Expected output:
```
GPU Verification Script for WSL2 RTX 5090
=========================================

============================================================
1. Checking JAX GPU Detection
============================================================
   GPU detected: cuda:0

============================================================
2. Checking JAX Default Backend
============================================================
   Default backend: gpu

============================================================
3. Running Matrix Multiply Test
============================================================
   Result shape: (1000, 1000), time: X.Xms

============================================================
4. Running NumPyro MCMC Test
============================================================
   Samples: mu=(200,), sigma=(200,), time: X.Xs

============================================================
5. Checking GPU Info (fit.py)
============================================================
   GPU: NVIDIA GeForce RTX 5090, <VRAM_MiB> MiB

============================================================
SUMMARY
============================================================

  [+] JAX GPU Detection: PASS
  [+] JAX Default Backend: PASS
  [+] Matrix Multiply Test: PASS
  [+] NumPyro MCMC Test: PASS
  [+] GPU Info (fit.py): PASS

SUCCESS: All GPU verification checks passed!
```

## Troubleshooting

### Problem: nvidia-smi not found in WSL2

**Cause:** Windows NVIDIA driver not installed or WSL2 kernel too old.

**Solution:**
1. Install/update Windows NVIDIA driver
2. Update WSL2: `wsl --update`
3. Restart WSL2: `wsl --shutdown` then reopen terminal

### Problem: "kernel version does not match DSO version"

**Cause:** Linux NVIDIA drivers were installed in WSL2 (breaks GPU passthrough).

**Solution:**
```bash
# Remove Linux drivers
sudo apt-get remove --purge nvidia-*
sudo apt-get autoremove

# Reinstall ONLY cuda-toolkit
sudo apt-get install cuda-toolkit-12-8
```

### Problem: "sm_120 not supported" or XLA compilation errors

**Cause:** JAX version too old for RTX 5090 (Blackwell architecture).

**Solution:**
```bash
pip install --upgrade "jax[cuda12]"
python -c "import jax; print(jax.__version__)"  # Should be >= 0.4.38
```

### Problem: JAX detects CPU only despite nvidia-smi working

**Cause:** Often LD_LIBRARY_PATH pointing to incompatible CUDA libraries.

**Solution:**
```bash
# Check if LD_LIBRARY_PATH is set
echo $LD_LIBRARY_PATH

# If set to CUDA paths, unset it (JAX bundles correct versions)
unset LD_LIBRARY_PATH

# Re-run verification
python scripts/verify_gpu.py
```

### Problem: MCMC runs but performance is slow

**Cause:** May be using CPU fallback without error, or suboptimal settings.

**Solution:**
1. Verify `jax.default_backend()` returns "gpu"
2. Monitor GPU usage: `watch -n1 nvidia-smi`
3. Check for divergences in MCMC output (may indicate model issues)

### Problem: "CUDA library version mismatch"

**Cause:** Multiple CUDA installations with conflicting versions.

**Solution:**
```bash
# Remove conflicting installations
pip uninstall jax jaxlib
sudo apt-get remove cuda-toolkit-* cuda-*

# Clean install
sudo apt-get install cuda-toolkit-12-8
pip install "jax[cuda12]"
```

## Known Limitations

### RTX 5090 (Blackwell) Random Number Non-Determinism

NVIDIA has documented that on consumer Blackwell GPUs (sm_120), the JAX random number generator may be non-deterministic even with fixed seeds. This is a hardware-level limitation.

**Impact:** Exact reproducibility of MCMC chains may vary between runs on RTX 5090. The statistical properties of the posterior remain valid, but specific sample values may differ.

**Mitigation:** For critical reproducibility requirements, consider running on non-Blackwell hardware or documenting the GPU used alongside results.

Reference: [NVIDIA JAX Release Notes](https://docs.nvidia.com/deeplearning/frameworks/jax-release-notes/)

### WSL2 Memory Limitations

WSL2 shares memory with Windows. For large MCMC jobs, ensure sufficient system RAM is available.

Configure WSL2 memory limit in `%USERPROFILE%\.wslconfig`:
```ini
[wsl2]
memory=32GB
```

## Performance Tips

1. **Use vectorized chain method:** MCMCConfig defaults to `chain_method='sequential'` for stability. For GPU efficiency, explicitly set `chain_method='vectorized'` when constructing MCMCConfig.
2. **Monitor GPU memory:** `nvidia-smi` during runs to check utilization
3. **Batch size:** Start with defaults; adjust if running out of GPU memory
4. **Close other GPU apps:** Games, browsers with hardware acceleration can compete for GPU

## References

- [JAX Installation Documentation](https://docs.jax.dev/en/latest/installation.html)
- [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [NumPyro MCMC Documentation](https://num.pyro.ai/en/stable/mcmc.html)
- [Microsoft WSL2 GPU Support](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)
