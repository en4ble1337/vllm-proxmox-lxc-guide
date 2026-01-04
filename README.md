# vLLM Installation Guide for Proxmox LXC with GPU Passthrough

![vLLM Infrastructure](infographic_flow.png)

## What is vLLM?

vLLM (Very Large Language Model) is a high-performance inference and serving engine for large language models. Think of it as a specialized server that makes AI models run faster and handle more users at the same time. It's designed to squeeze the maximum performance out of your GPU, making it ideal for running AI chatbots, text generation services, or any application that needs to process many requests quickly.

### Why Use vLLM?

vLLM offers several key advantages that make it a popular choice for deploying AI models:

- **Faster Response Times**: Uses advanced memory management (PagedAttention) to serve requests 2-24x faster than traditional methods
- **More Concurrent Users**: Handles many users simultaneously without slowing down, increasing throughput by up to 24x
- **Easy to Use**: Compatible with OpenAI API format, so existing code works without changes
- **Memory Efficient**: Smart memory management allows serving larger models or more users with the same hardware
- **Production Ready**: Supports continuous batching, streaming responses, and enterprise features out of the box

### vLLM vs llama.cpp: Which Should You Choose?

Both are excellent tools for running AI models locally, but they serve different needs:

| Feature | vLLM | llama.cpp |
|---------|------|-----------|
| **Best For** | High-throughput servers, multiple concurrent users | Single-user desktop applications, CPU inference |
| **Hardware Focus** | NVIDIA GPUs (CUDA required) | CPU-first, optional GPU support |
| **Performance** | 2-24x faster throughput for batch requests | Optimized for low latency single requests |
| **Memory Management** | PagedAttention (efficient batching) | Traditional KV-cache management |
| **API Compatibility** | OpenAI API compatible | Simple HTTP API or CLI |
| **Setup Complexity** | Moderate (requires CUDA, Python environment) | Simple (single binary, minimal dependencies) |
| **Quantization** | Limited (mainly FP16, BF16) | Extensive (GGUF format with 2-8 bit options) |
| **Use Case** | Production API servers, multi-user chat services | Personal assistants, desktop apps, resource-constrained devices |
| **Model Support** | Wide (HuggingFace transformers) | Primarily GGUF format models |

**Choose vLLM if:** You need to serve many users simultaneously, want maximum GPU utilization, or are building a production API service.

**Choose llama.cpp if:** You want simple setup, need CPU inference, run models on lower-end hardware, or prefer heavily quantized models (2-4 bit).

---

## Overview

This guide installs vLLM with flash-attention on a Proxmox LXC container with GPU passthrough. 
> [!CAUTION]
> CUDA toolkit must be installed on **BOTH** the Proxmox host AND the LXC container. This is commonly missed and causes flash-attn compilation failures (segfaults).

---

## Prerequisites

Before starting this guide, ensure you have:

1. **GPU Passthrough Configured**: Follow the [GPU Passthrough for Proxmox LXC Container](https://github.com/en4ble1337/GPU-Passthrough-for-Proxmox-LXC-Container) guide to set up your Proxmox environment for GPU access
2. **NVIDIA GPU**: Compatible NVIDIA GPU with at least 8GB VRAM (see compatibility matrix below)
3. **Proxmox VE 8.4+**: Running on Debian 12 base
4. **Basic Linux Knowledge**: Comfortable with command line and SSH

---

## Environment Reference

| Component | Value |
|-----------|-------|
| Proxmox VE | 8.4+ |
| Proxmox Base | Debian 12 |
| LXC OS | Ubuntu 22.04 |
| GPU | NVIDIA RTX 3080 (10GB VRAM) |
| NVIDIA Driver | 580.x (on host AND LXC) |
| CUDA Toolkit | 12.8 (on host AND LXC) |
| Python | 3.10.x |
| Package Manager | python venv + pip |

---

## Compatibility Matrices

### NVIDIA Driver ↔ CUDA Toolkit

| Driver Version | Max CUDA Version | Notes |
|----------------|------------------|-------|
| 550.x | 12.4 | Requires `--index-url .../cu124` for PyTorch |
| 555.x | 12.5 | |
| 560.x | 12.6 | |
| 565.x | 12.7 | |
| 570.x | 12.8 | |
| 580.x | 12.8 | **Recommended** - matches PyTorch defaults |

### GPU Architecture (Compute Capability)

| GPU Series | Architecture | Compute Capability |
|------------|--------------|-------------------|
| GTX 10xx | Pascal | 6.1 |
| RTX 20xx | Turing | 7.5 |
| RTX 30xx | Ampere | 8.6 |
| RTX 40xx | Ada Lovelace | 8.9 |
| RTX 50xx | Blackwell | 12.0 |
| A100 | Ampere | 8.0 |
| H100 | Hopper | 9.0 |

---

# PART 1: PROXMOX HOST SETUP

> [!CAUTION]
> Do not skip this section! Flash-attn will segfault without host CUDA.**

These steps are performed on the **Proxmox host**, not inside an LXC.

Access via: Proxmox web UI → Select node → Shell (or SSH to host)

---

## Phase 0A: Install NVIDIA Driver on Host

> [!NOTE]
> For detailed GPU passthrough setup, see the [GPU Passthrough for Proxmox LXC Container](https://github.com/en4ble1337/GPU-Passthrough-for-Proxmox-LXC-Container) guide.

Skip if already installed. Verify with `nvidia-smi`.

```bash
# Download latest production driver
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/580.76.05/NVIDIA-Linux-x86_64-580.76.05.run

# Make executable and install
chmod +x NVIDIA-Linux-x86_64-580.76.05.run
./NVIDIA-Linux-x86_64-580.76.05.run

# Verify
nvidia-smi
```

**Note:** For RTX 50xx (Blackwell) GPUs, select MIT drivers instead of Proprietary during installation.

---

## Phase 0B: Install Prerequisites on Host

```bash
apt install -y \
    g++ \
    freeglut3-dev \
    build-essential \
    libx11-dev \
    libxmu-dev \
    libxi-dev \
    libglu1-mesa-dev \
    libfreeimage-dev \
    libglfw3-dev \
    wget \
    htop \
    btop \
    nvtop \
    glances \
    git \
    pciutils \
    cmake \
    curl \
    libcurl4-openssl-dev
```

---

## Phase 0C: Install CUDA Toolkit 12.8 on Host

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update
apt install -y cuda-toolkit-12-8
```

**Note:** CUDA toolkit installation may break NVIDIA driver. If `nvidia-smi` stops working after this step, reinstall the driver (Phase 0A).

---

## Phase 0D: Configure CUDA PATH on Host

```bash
cp ~/.bashrc ~/.bashrc-backup
echo 'export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
source ~/.bashrc
```

---

## Phase 0E: Reboot Host and Verify

```bash
reboot now
```

After reboot, verify:

```bash
nvidia-smi
nvcc --version
```

**Expected:** 
- nvidia-smi shows driver 580.x, CUDA 12.8
- nvcc shows `Cuda compilation tools, release 12.8`

> [!IMPORTANT] 
> **If nvidia-smi fails:** Reinstall NVIDIA driver (Phase 0A), then verify again.

**STOP if either command fails. Do not proceed to LXC setup.**

---

# PART 2: LXC CONTAINER SETUP

These steps are performed **inside the LXC container**.

---

## Phase 1: Create LXC with GPU Passthrough

Create Ubuntu 22.04 LXC in Proxmox with these settings:

**Resources:**
- CPU: 4+ cores
- RAM: 8GB minimum (28GB+ if building flash-attn from source)
- Disk: 50GB+

**LXC Config** (`/etc/pve/lxc/YOUR_VMID.conf`):

```
arch: amd64
cores: 8
memory: 8192
ostype: ubuntu
unprivileged: 1
features: nesting=1

# GPU Passthrough
lxc.cgroup2.devices.allow: c 195:* rwm
lxc.cgroup2.devices.allow: c 509:* rwm
lxc.cgroup2.devices.allow: c 510:* rwm
lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file
lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-modeset dev/nvidia-modeset none bind,optional,create=file
```

For multiple GPUs, add additional nvidia entries (nvidia1, nvidia2, etc.)

> [!TIP]
> **Detailed Instructions:** See [GPU Passthrough for Proxmox LXC Container](https://github.com/en4ble1337/GPU-Passthrough-for-Proxmox-LXC-Container) for step-by-step GPU passthrough configuration.

---

## Phase 2: Install NVIDIA Driver in LXC

From **Proxmox host**, push driver to LXC:

```bash
# On host - push driver file to LXC
pct push YOUR_VMID NVIDIA-Linux-x86_64-580.76.05.run /root/NVIDIA-Linux-x86_64-580.76.05.run
```

Then inside LXC:

```bash
chmod +x NVIDIA-Linux-x86_64-580.76.05.run
./NVIDIA-Linux-x86_64-580.76.05.run --no-kernel-modules
```

**Important:** Use `--no-kernel-modules` flag inside LXC.

Verify:

```bash
nvidia-smi
```

---

## Phase 3: Install Prerequisites in LXC

```bash
apt update
apt upgrade -y
apt install -y \
    g++ \
    freeglut3-dev \
    build-essential \
    libx11-dev \
    libxmu-dev \
    libxi-dev \
    libglu1-mesa-dev \
    libfreeimage-dev \
    libglfw3-dev \
    wget \
    htop \
    btop \
    git \
    pciutils \
    cmake \
    curl \
    libcurl4-openssl-dev \
    python3-full \
    python3-pip \
    python3-venv
```

---

## Phase 4: Install CUDA Toolkit 12.8 in LXC

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update
apt install -y cuda-toolkit-12-8
```

Note: Some optional packages may fail — that's OK if nvcc works.

> [!NOTE]
> If `nvidia-smi` stops working after this step, reinstall the driver (Phase 2).

---

## Phase 5: Configure CUDA PATH in LXC

```bash
cp ~/.bashrc ~/.bashrc-backup
echo 'export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
nvidia-smi
nvcc --version
```

**Expected:** 
- nvidia-smi shows GPU
- nvcc shows `Cuda compilation tools, release 12.8`

---

## Phase 6: Create Python Environment

```bash
mkdir ~/vllm-project
cd ~/vllm-project
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now show `(.venv)`.

---

## Phase 7: Install vLLM and Dependencies

```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
pip install flash-attn
pip install transformers
pip install huggingface_hub
```

> [!NOTE]
> With CUDA toolkit on both host and LXC, `pip install flash-attn` should work without issues.

---

## Phase 8: Configure HuggingFace

```bash
huggingface-cli login
```
> [!NOTE]
> Need to register with huggingface and get your API
---

## Phase 9: Test vLLM

```bash
python3 << 'EOF'
from vllm import LLM, SamplingParams

print("Loading TinyLlama-1.1B...")

llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    gpu_memory_utilization=0.8,
    dtype="half"
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = llm.generate(["Tell me a joke about programming."], sampling_params)

print(f"\nResponse: {outputs[0].outputs[0].text.strip()}")
print("\nSUCCESS: vLLM is working!")
EOF
```

---

## Phase 10: Verify Flash-Attention

```bash
cat << 'SCRIPT' > vllm_test_flash.py
# Check flash-attn is installed
try:
    import flash_attn
    print(f"✓ flash-attn version: {flash_attn.__version__}")
except ImportError:
    print("✗ flash-attn: NOT INSTALLED")
    exit(1)

# Check PyTorch CUDA
import torch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# Quick functional test
from flash_attn import flash_attn_func
import torch

# Create test tensors
batch, heads, seq_len, head_dim = 1, 8, 64, 64
q = torch.randn(batch, seq_len, heads, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn(batch, seq_len, heads, head_dim, device='cuda', dtype=torch.float16)
v = torch.randn(batch, seq_len, heads, head_dim, device='cuda', dtype=torch.float16)

# Run flash attention
output = flash_attn_func(q, k, v)
print(f"✓ flash-attn functional test: PASSED (output shape: {output.shape})")
print("\n✓ Flash-attention is working correctly!")
SCRIPT

python3 vllm_test_flash.py
```

**Expected output:**
```
✓ flash-attn version: 2.x.x
✓ PyTorch version: 2.x.x+cu128
✓ CUDA available: True
✓ GPU: NVIDIA GeForce RTX 3080
✓ flash-attn functional test: PASSED
✓ Flash-attention is working correctly!
```

---

## Phase 11: Start API Server

```bash
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key your-api-key \
    --gpu-memory-utilization 0.8 \
    --dtype half
```

Test from another machine:

```bash
curl -s http://YOUR_LXC_IP:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer your-api-key" \
    -d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 50}' \
    | jq '.choices[0].message.content'
```

---

## Phase 12: Create Snapshot

In Proxmox, snapshot this working state before making changes:

Datacenter → Your LXC → Snapshots → Take Snapshot

---

## Startup Script

```bash
cat > ~/start-vllm.sh << 'SCRIPT'
#!/bin/bash
cd ~/vllm-project
source .venv/bin/activate

MODEL="${1:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
PORT="${2:-8000}"
API_KEY="${3:-changeme}"

echo "Starting vLLM: $MODEL on port $PORT"

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --gpu-memory-utilization 0.85 \
    --dtype half
SCRIPT

chmod +x ~/start-vllm.sh
```

Usage:

```bash
~/start-vllm.sh "model-name" 8000 "api-key"
```

---

## Troubleshooting

### flash-attn segfault during compilation

**Cause:** CUDA toolkit not installed on Proxmox host.

**Fix:** Complete Part 1 (Proxmox Host Setup), reboot host, then retry.

### nvidia-smi stops working after CUDA toolkit install

**Cause:** CUDA toolkit can break driver installation.

**Fix:** Reinstall NVIDIA driver:
- On host: `./NVIDIA-Linux-x86_64-580.76.05.run`
- In LXC: `./NVIDIA-Linux-x86_64-580.76.05.run --no-kernel-modules`

### pipenv errors (ImportError, urllib3)

**Cause:** System pipenv is broken on some Ubuntu versions.

**Fix:** Use python venv instead:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### flash-attn OOM during compilation

If building from source with limited RAM:

```bash
export TORCH_CUDA_ARCH_LIST="8.6"  # Your GPU's compute capability
export MAX_JOBS=1
pip install flash-attn --no-build-isolation
```

| RAM Available | MAX_JOBS Setting |
|---------------|------------------|
| 16GB | Skip flash-attn |
| 24GB | MAX_JOBS=1 |
| 32GB | MAX_JOBS=2 |
| 64GB+ | Default |

### pip hash mismatch errors

Use `--no-cache-dir`:

```bash
pip install --no-cache-dir vllm --extra-index-url https://download.pytorch.org/whl/cu128
```

### nvcc not found

```bash
source ~/.bashrc
# or
export PATH=/usr/local/cuda-12.8/bin:$PATH
```

### CUDA out of memory at runtime

```bash
--gpu-memory-utilization 0.7
```

### "Device string must not be empty" error

**Cause:** PyTorch can't see GPU.

**Fix:** 
1. Check `nvidia-smi` works
2. If not, reinstall NVIDIA driver
3. Verify: `python3 -c "import torch; print(torch.cuda.is_available())"`

---

## Quick Reference

```bash
# Activate environment
cd ~/vllm-project && source .venv/bin/activate

# Check GPU
nvidia-smi

# Check CUDA
nvcc --version

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Check flash-attn
python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')"

# Start server
~/start-vllm.sh
```

---

## Summary Checklist

### Host Setup (CRITICAL!)
- [ ] NVIDIA driver 580.x installed on host
- [ ] Prerequisites installed on host
- [ ] CUDA toolkit 12.8 installed on host
- [ ] PATH configured on host
- [ ] Host rebooted
- [ ] `nvidia-smi` works on host
- [ ] `nvcc --version` works on host

### LXC Setup
- [ ] LXC created with GPU passthrough config
- [ ] NVIDIA driver installed in LXC (with `--no-kernel-modules`)
- [ ] `nvidia-smi` works in LXC
- [ ] Prerequisites installed in LXC
- [ ] CUDA toolkit 12.8 installed in LXC
- [ ] `nvcc --version` works in LXC
- [ ] Python venv created
- [ ] vLLM installed
- [ ] flash-attn installed
- [ ] flash-attn verification passed
- [ ] HuggingFace authenticated
- [ ] vLLM test passed
- [ ] Snapshot created

---

## Version History

| Version | Changes |
|---------|---------|
| v12 | Added vLLM introduction, comparison table, GitHub formatting, prerequisite links, updated flash-attn test |
| v11 | Fixed: Use python venv instead of pipenv, added flash-attn verification test, driver reinstall notes |
| v10 | Added Proxmox host CUDA setup |
| v9 | Git clone flash-attn + setup.py install |
| v8 | Added compatibility matrices, TORCH_CUDA_ARCH_LIST |
| v7 | Install order fix for CUDA 12.4 |

---

## Key Lessons Learned

| Problem | Root Cause | Solution |
|---------|------------|----------|
| flash-attn segfault | **CUDA toolkit missing on Proxmox host** | Install CUDA on host AND LXC |
| nvidia-smi breaks after CUDA install | CUDA toolkit overwrites driver files | Reinstall NVIDIA driver |
| pipenv ImportError | Broken system pipenv on Ubuntu | Use python venv instead |
| flash-attn OOM | Multi-arch parallel compilation | Set TORCH_CUDA_ARCH_LIST + MAX_JOBS=1 |
| pip hash errors | pip 25.x bug | Use --no-cache-dir |
| "Device string must not be empty" | PyTorch can't see GPU | Reinstall NVIDIA driver |

---

## References

### Project Resources
- [vLLM Official Documentation](https://docs.vllm.ai/)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [GPU Passthrough for Proxmox LXC Container](https://github.com/en4ble1337/GPU-Passthrough-for-Proxmox-LXC-Container)

### Community Guides
- [Digital Spaceport - Ollama + OpenWebUI Setup](https://digitalspaceport.com/how-to-setup-an-ai-server-homelab-beginners-guides-ollama-and-openwebui-on-proxmox-lxc/)
- [Digital Spaceport - Llama.cpp Setup](https://digitalspaceport.com/llama-cpp-on-proxmox-9-lxc-how-to-setup-an-ai-server-homelab-beginners-guides/)
- [Digital Spaceport - vLLM Setup](https://digitalspaceport.com/how-to-setup-vllm-local-ai-homelab-ai-server-beginners-guides/)

### Technical Documentation
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [Flash-Attention GitHub](https://github.com/Dao-AILab/flash-attention)
- [PyTorch CUDA Compatibility](https://pytorch.org/get-started/locally/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

### Related Tools
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - CPU-focused LLM inference
- [Ollama](https://ollama.ai/) - Easy local LLM deployment
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui) - Web interface for LLMs

---

## License

This guide is provided as-is for educational purposes. Please refer to individual project licenses for vLLM, CUDA, and other mentioned tools.

---

## Contributing

Found an issue or have improvements? Please open an issue or pull request on GitHub.

---

**Last Updated:** January 2026
