# Getting Started

Step-by-step guide from cloning the repository to running your first prediction.

## Prerequisites

- Python >= 3.11
- NVIDIA GPU (optional but recommended for MCMC)
- Git

## Step 1: Clone the Repository

```bash
git clone https://github.com/cupidthatbtc/panelcast.git
cd panelcast
```

## Step 2: Install pixi Package Manager

pixi handles all Python dependencies automatically.

**Linux/macOS:**
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

**Windows (PowerShell):**
```powershell
iwr -useb https://pixi.sh/install.ps1 | iex
```

After installation, restart your terminal or run `source ~/.bashrc` (Linux/macOS).

## Step 3: Install Dependencies

```bash
pixi install
```

This creates a `.pixi` environment with all required packages including JAX, NumPyro, and pandas.

## Step 4: Prepare Your Dataset

Your CSV file needs these columns:

**Required columns:**
- `Artist` - Artist name
- `Album` - Album title
- `Year` - Release year
- `Release Date` - Full release date
- `Genres` - Pipe-separated genres (e.g., "Pop|Rock")
- `User Score` - Target variable (0-100)
- `User Ratings` - Number of user ratings
- `Tracks` - Number of tracks
- `Runtime (min)` - Album runtime
- `Avg Track Runtime (min)` - Average track length
- `Album Type` - LP, EP, Mixtape, etc.
- `All Artists` - Pipe-separated list of all artists

**Optional columns:**
- `Critic Score`, `Critic Reviews`, `Avg Track Score`
- `Descriptors`, `Label`, `Album URL`

See [DATA_CONTRACT.md](DATA_CONTRACT.md) for full schema details.

## Step 5: Configure Dataset Path

Choose one of these options:

**Option A: Environment variable (recommended)**

Linux/macOS:
```bash
export AOTY_DATASET_PATH="/path/to/your/data.csv"
```

Windows PowerShell:
```powershell
$env:AOTY_DATASET_PATH = "C:\path\to\your\data.csv"
```

Windows CMD:
```cmd
set AOTY_DATASET_PATH=C:\path\to\data.csv
```

**Option B: Create .env file**

```bash
cp .env.example .env
# Edit .env and set your path
```

## Step 6: (Optional) GPU Setup

GPU acceleration significantly speeds up MCMC sampling. If you have an NVIDIA GPU:

1. Ensure Windows NVIDIA drivers are installed (WSL2 uses passthrough)
2. Verify GPU access:
   ```bash
   python scripts/verify_gpu.py
   ```

See [GPU_SETUP.md](GPU_SETUP.md) for detailed WSL2/CUDA configuration.

## Step 7: Verify Installation

```bash
# Show available commands
panelcast --help

# Check GPU memory and data loading (no actual training)
panelcast run --preflight-only
```

If preflight passes (exit code 0), you're ready to run.

## Step 8: Run the Pipeline

**Quick exploratory run** (~5-10 min on GPU):
```bash
panelcast run --num-chains 1 --num-samples 500
```

**Full run** (default settings, ~30-60 min on GPU):
```bash
panelcast run
```

**Quality run** (~2-4 hours on GPU):
```bash
panelcast run --num-chains 8 --num-samples 2000 --target-accept 0.95 --strict
```

## What Happens Next

After a successful run, outputs are saved to:

| Directory | Contents |
|-----------|----------|
| `outputs/` | Run manifests and runtime artifacts (`outputs/<run_id>/manifest.json`) |
| `outputs/evaluation/` | Evaluation metrics, diagnostics, predictions, calibration payloads |
| `reports/` | Publication artifacts (figures, tables, model cards) |
| `data/processed/` | Cleaned datasets |
| `data/features/` | Feature matrices and feature manifest |
| `data/splits/` | Train/validation/test split manifests |

To regenerate reports or run individual stages:
```bash
panelcast stage report
panelcast stage evaluate
```

See [CLI.md](CLI.md) for the complete command reference.

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| `AOTY_DATASET_PATH not set` | Set the environment variable or create `.env` file |
| `nvidia-smi: command not found` | Install NVIDIA Windows driver, then `wsl --update` |
| JAX using CPU despite GPU present | Unset `LD_LIBRARY_PATH` and retry |
| Preflight shows insufficient memory | Use `--num-chains 1` or reduce `--num-samples` |
| MCMC divergences | Try `--num-warmup 2000` or `--target-accept 0.95` |
| `pixi: command not found` | Restart terminal after pixi installation |

For detailed troubleshooting:
- GPU issues: [GPU_SETUP.md](GPU_SETUP.md)
- CLI options: [CLI.md](CLI.md)
- Data format: [DATA_CONTRACT.md](DATA_CONTRACT.md)
