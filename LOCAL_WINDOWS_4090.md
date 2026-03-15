# Local Windows / RTX 4090 Setup Notes

This document summarizes what was built for running nanochat locally on Windows with a single RTX 4090-class GPU, what blocked execution, how those issues were fixed, how to run the new workflow, and what to do next.

## What was added

### 1. Windows-local runner

Added [runs/run4090.ps1](runs/run4090.ps1).

Purpose:

- provide a Windows-native PowerShell entrypoint instead of relying on the Linux bash scripts in `runs/*.sh`
- use settings that make sense for a single local GPU instead of an 8xH100 box
- expose staged execution so setup, tokenizer training, base training, SFT, evaluation, CLI chat, and web chat can be run separately

Supported stages:

- `setup`
- `tokenizer`
- `train`
- `eval`
- `sft`
- `chat-cli`
- `chat-web`
- `all`

Default local training settings in the script:

- depth `d6`
- `--window-pattern L`
- `--device-type cuda`
- no `--fp8`
- `NANOCHAT_DISABLE_COMPILE=1`

### 2. README update

Added a local Windows / RTX 4090 section to [README.md](README.md) pointing users to [runs/run4090.ps1](runs/run4090.ps1) instead of the H100-oriented [runs/speedrun.sh](runs/speedrun.sh).

### 3. Compile guard for Windows execution

Added an opt-out for `torch.compile` in:

- [scripts/base_train.py](scripts/base_train.py)
- [scripts/chat_sft.py](scripts/chat_sft.py)

These now respect `NANOCHAT_DISABLE_COMPILE=1` and run in eager mode when needed.

### 4. Optimizer fused-kernel compile guard

Added an opt-out in [nanochat/optim.py](nanochat/optim.py) so the fused AdamW and Muon step functions do not require `torch.compile` when `NANOCHAT_DISABLE_COMPILE=1` is set.

This was necessary because disabling compile on the model alone was not enough; the optimizer still tried to go through Triton/Inductor.

## Blockers we hit

### 1. `uv` was unusable on this machine

Observed behavior:

- calling `uv` failed immediately with a pyenv error asking for `pyenv global` / `pyenv local`

Root cause:

- PowerShell resolved `uv` to a pyenv shim at `C:\Users\ragha\.pyenv\pyenv-win\shims\uv.bat`
- there was no usable underlying `uv` binary configured through pyenv

Fix:

- updated [runs/run4090.ps1](runs/run4090.ps1) so setup first tries `uv`, but falls back to direct `pip` installation in the existing `.venv` if `uv` fails

### 2. Torch wheel download failed during setup

Observed behavior:

- first install attempt for `torch==2.9.1+cu128` failed with a network reset while downloading the ~2.9 GB wheel

Root cause:

- transient network interruption during pip download

Fix:

- retried installation with higher retries and timeout
- torch then installed successfully and CUDA became visible to PyTorch

### 3. `pip install -e .` failed in this repo layout

Observed behavior:

- editable install failed with setuptools package discovery errors about multiple top-level packages in a flat layout

Root cause:

- this repository is not packaged in a way that setuptools editable install accepted in this environment

Fix:

- changed the fallback setup path in [runs/run4090.ps1](runs/run4090.ps1) to install the runtime dependency list directly instead of doing `pip install -e .`

### 4. `torch.compile` failed due missing Triton on Windows

Observed behavior:

- base training started, then crashed with `torch._inductor.exc.TritonMissing`

Root cause:

- Windows environment did not have a working Triton setup for the compiled model path

Fix:

- added `NANOCHAT_DISABLE_COMPILE=1` handling in [scripts/base_train.py](scripts/base_train.py)
- added the same handling in [scripts/chat_sft.py](scripts/chat_sft.py)
- set `NANOCHAT_DISABLE_COMPILE=1` by default inside [runs/run4090.ps1](runs/run4090.ps1)

### 5. Optimizer still required compile even after model compile was disabled

Observed behavior:

- after disabling `torch.compile` on the model, training still crashed during optimizer step with the same Triton/Inductor error

Root cause:

- the fused optimizer functions in [nanochat/optim.py](nanochat/optim.py) were also decorated with `torch.compile`

Fix:

- added `_maybe_compile` wrapper in [nanochat/optim.py](nanochat/optim.py)
- fused optimizer kernels now skip compile when `NANOCHAT_DISABLE_COMPILE=1`

## What was successfully run

### Environment validation

Confirmed in the local `.venv`:

- `torch 2.9.1+cu128`
- CUDA available
- GPU detected as `NVIDIA GeForce RTX 4090 Laptop GPU`

### Tokenizer stage

Ran the tokenizer stage successfully.

Artifacts created under the local cache dir:

- `C:\Users\ragha\.cache\nanochat\tokenizer\tokenizer.pkl`
- `C:\Users\ragha\.cache\nanochat\tokenizer\token_bytes.pt`

### Base training smoke test

Ran a 2-iteration base training smoke test successfully.

Key outcomes:

- model initialized on GPU
- validation ran
- training steps completed
- checkpoint and optimizer state were saved

Artifacts created:

- `C:\Users\ragha\.cache\nanochat\base_checkpoints\d6\model_000002.pt`
- `C:\Users\ragha\.cache\nanochat\base_checkpoints\d6\meta_000002.json`
- `C:\Users\ragha\.cache\nanochat\base_checkpoints\d6\optim_000002_rank0.pt`

Representative runtime facts from the successful smoke run:

- single GPU world size: 1
- compute dtype: bf16
- Flash Attention 3 unavailable, using SDPA fallback
- compile disabled intentionally via `NANOCHAT_DISABLE_COMPILE=1`
- peak VRAM observed in smoke run: about 3.2 GiB

## How to run the workflow

All commands below are intended to be run from the repo root in PowerShell.

### 1. Setup

```powershell
powershell -ExecutionPolicy Bypass -File .\runs\run4090.ps1 -Stage setup
```

What this does:

- ensures `.venv` exists
- tries `uv sync --extra gpu`
- falls back to pip if `uv` is broken on this machine
- validates CUDA visibility in PyTorch

### 2. Train the tokenizer

```powershell
powershell -ExecutionPolicy Bypass -File .\runs\run4090.ps1 -Stage tokenizer
```

What this does:

- downloads the initial data shards
- trains tokenizer artifacts into the nanochat cache dir
- runs tokenizer evaluation

### 3. Base pretraining

Default local run:

```powershell
powershell -ExecutionPolicy Bypass -File .\runs\run4090.ps1 -Stage train
```

Short smoke test:

```powershell
powershell -ExecutionPolicy Bypass -File .\runs\run4090.ps1 -Stage train -TrainIterations 2
```

Longer local experiment:

```powershell
powershell -ExecutionPolicy Bypass -File .\runs\run4090.ps1 -Stage train -TrainIterations 3000
```

Useful overrides:

```powershell
powershell -ExecutionPolicy Bypass -File .\runs\run4090.ps1 -Stage train -Depth 12 -DeviceBatchSize 2
powershell -ExecutionPolicy Bypass -File .\runs\run4090.ps1 -Stage train -Depth 12 -DeviceBatchSize 1
powershell -ExecutionPolicy Bypass -File .\runs\run4090.ps1 -Stage train -ForceFloat32
```

### 4. Base checkpoint evaluation

```powershell
powershell -ExecutionPolicy Bypass -File .\runs\run4090.ps1 -Stage eval
```

### 5. Supervised fine-tuning

```powershell
powershell -ExecutionPolicy Bypass -File .\runs\run4090.ps1 -Stage sft
```

What this does:

- downloads `identity_conversations.jsonl` if needed
- loads the base checkpoint for the current model tag
- runs SFT with local single-GPU defaults

### 6. Chat in CLI

```powershell
powershell -ExecutionPolicy Bypass -File .\runs\run4090.ps1 -Stage chat-cli
```

Custom prompt:

```powershell
powershell -ExecutionPolicy Bypass -File .\runs\run4090.ps1 -Stage chat-cli -Prompt "Why is the sky blue?"
```

### 7. Chat in browser

```powershell
powershell -ExecutionPolicy Bypass -File .\runs\run4090.ps1 -Stage chat-web
```

Then open:

- `http://127.0.0.1:8000`

### 8. Run multiple stages in sequence

```powershell
powershell -ExecutionPolicy Bypass -File .\runs\run4090.ps1 -Stage setup,tokenizer,train
powershell -ExecutionPolicy Bypass -File .\runs\run4090.ps1 -Stage sft
powershell -ExecutionPolicy Bypass -File .\runs\run4090.ps1 -Stage chat-web
```

## Why the local defaults differ from the repo speedrun

The official [runs/speedrun.sh](runs/speedrun.sh) is designed for Linux on 8xH100. That is not a good default for a local Windows machine.

The Windows-local runner deliberately changes the following:

- uses PowerShell instead of bash
- does not use `--fp8`
- uses `--window-pattern L`
- runs single-GPU Python entrypoints directly instead of distributed `torchrun`
- disables compile by default because the Windows runtime here did not have a working Triton path

## Interpreting the smoke-test training output

The successful 2-step run was only a smoke test. It proves the system can train, but it does not produce a meaningful model.

Important implications:

- nonsense text samples after 2 steps are expected
- small improvement in validation bpb is enough to show the loop is functioning
- saved checkpoints are the main success criterion for that test

## What to do next

### Recommended next step

Run a longer base training job with the current default `d6` setup first:

```powershell
powershell -ExecutionPolicy Bypass -File .\runs\run4090.ps1 -Stage train -TrainIterations 3000
```

This is the safest next move because:

- tokenizer is already trained
- environment is already validated
- the training path has already been smoke-tested end to end

### After that

1. Run SFT:

```powershell
powershell -ExecutionPolicy Bypass -File .\runs\run4090.ps1 -Stage sft
```

2. Open the web UI:

```powershell
powershell -ExecutionPolicy Bypass -File .\runs\run4090.ps1 -Stage chat-web
```

3. If you want a larger model, increase depth gradually:

```powershell
powershell -ExecutionPolicy Bypass -File .\runs\run4090.ps1 -Stage train -Depth 12 -DeviceBatchSize 2
```

4. If you hit VRAM limits, reduce `-DeviceBatchSize` to `2` or `1`.

## Current status summary

- local Windows runner exists and works
- setup works on this machine despite broken `uv`
- tokenizer stage works
- base training works on the 4090 with compile disabled
- checkpoints are being written successfully
- next untested stage in this specific environment is a full SFT run and then browser chat on top of that checkpoint