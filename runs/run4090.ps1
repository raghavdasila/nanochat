param(
    [ValidateSet("setup", "tokenizer", "train", "eval", "sft", "chat-cli", "chat-web", "all")]
    [string[]]$Stage = @("setup"),
    [string]$BaseDir = "$HOME\.cache\nanochat",
    [int]$Depth = 6,
    [int]$HeadDim = 64,
    [int]$MaxSeqLen = 1024,
    [int]$DeviceBatchSize = 4,
    [int]$TotalBatchSize = 32768,
    [int]$TrainIterations = 3000,
    [int]$SftIterations = 1000,
    [string]$Run = "dummy",
    [string]$ModelTag = "",
    [int]$Port = 8000,
    [string]$Prompt = "Why is the sky blue?",
    [switch]$ForceFloat32
)

$ErrorActionPreference = "Stop"

function Write-Step($Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Require-Command($Name, $InstallHint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found. $InstallHint"
    }
}

function Setup-With-PipFallback() {
    Write-Warning "uv setup failed on this machine, falling back to pip in the existing .venv"
    Run-Python @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    Run-Python @(
        "-m", "pip", "install",
        "--index-url", "https://download.pytorch.org/whl/cu128",
        "torch==2.9.1"
    )
    Run-Python @(
        "-m", "pip", "install",
        "datasets>=4.0.0",
        "fastapi>=0.117.1",
        "ipykernel>=7.1.0",
        "kernels>=0.11.7",
        "matplotlib>=3.10.8",
        "numpy",
        "psutil>=7.1.0",
        "python-dotenv>=1.2.1",
        "regex>=2025.9.1",
        "rustbpe>=0.1.0",
        "scipy>=1.15.3",
        "tabulate>=0.9.0",
        "tiktoken>=0.11.0",
        "tokenizers>=0.22.0",
        "transformers>=4.57.3",
        "uvicorn>=0.36.0",
        "wandb>=0.21.3",
        "zstandard>=0.25.0"
    )
}

function Get-PythonPath() {
    $pythonPath = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
    $resolvedPath = Resolve-Path $pythonPath -ErrorAction SilentlyContinue
    if (-not $resolvedPath) {
        throw "Python environment not found at .venv. Run this script with -Stage setup first."
    }
    return $resolvedPath.Path
}

function Run-External($FilePath, [string[]]$Arguments) {
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

function Run-Python([string[]]$Arguments) {
    $pythonPath = Get-PythonPath
    Run-External $pythonPath $Arguments
}

function Ensure-IdentityConversations() {
    $identityPath = Join-Path $env:NANOCHAT_BASE_DIR "identity_conversations.jsonl"
    if (-not (Test-Path $identityPath)) {
        Write-Step "Downloading identity conversations"
        Invoke-WebRequest `
            -Uri "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl" `
            -OutFile $identityPath
    }
}

if (-not $ModelTag) {
    $ModelTag = "d$Depth"
}

$env:OMP_NUM_THREADS = "1"
$env:NANOCHAT_BASE_DIR = $BaseDir
$env:NANOCHAT_DISABLE_COMPILE = "1"
if ($ForceFloat32) {
    $env:NANOCHAT_DTYPE = "float32"
}

New-Item -ItemType Directory -Force -Path $env:NANOCHAT_BASE_DIR | Out-Null

$requestedStages = [System.Collections.Generic.List[string]]::new()
foreach ($value in $Stage) {
    if ($value -eq "all") {
        foreach ($expanded in @("setup", "tokenizer", "train", "eval", "sft", "chat-cli", "chat-web")) {
            $requestedStages.Add($expanded)
        }
    }
    else {
        $requestedStages.Add($value)
    }
}

if ($requestedStages.Contains("setup")) {
    Write-Step "Setting up uv environment and GPU dependencies"
    $venvPath = Join-Path $PSScriptRoot "..\.venv"
    if (-not (Test-Path $venvPath)) {
        Write-Step "Creating local virtual environment"
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            try {
                Run-External "uv" @("venv")
            }
            catch {
                py -3 -m venv "$venvPath"
            }
        }
        else {
            py -3 -m venv "$venvPath"
        }
    }

    if (Get-Command uv -ErrorAction SilentlyContinue) {
        try {
            Run-External "uv" @("sync", "--extra", "gpu")
        }
        catch {
            Setup-With-PipFallback
        }
    }
    else {
        Setup-With-PipFallback
    }

    Run-Python @(
        "-c",
        "import torch; assert torch.cuda.is_available(), 'CUDA is not available to PyTorch'; print(f'CUDA OK: {torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}')"
    )
}

if ($requestedStages.Contains("tokenizer")) {
    Write-Step "Downloading initial dataset shards and training tokenizer"
    Run-Python @("-m", "nanochat.dataset", "-n", "8")
    Run-Python @("-m", "scripts.tok_train", "--max-chars", "2000000000")
    Run-Python @("-m", "scripts.tok_eval")
}

if ($requestedStages.Contains("train")) {
    Write-Step "Running single-GPU base pretraining tuned for a local RTX 4090"
    Run-Python @(
        "-m", "scripts.base_train",
        "--device-type", "cuda",
        "--depth", "$Depth",
        "--head-dim", "$HeadDim",
        "--window-pattern", "L",
        "--max-seq-len", "$MaxSeqLen",
        "--device-batch-size", "$DeviceBatchSize",
        "--total-batch-size", "$TotalBatchSize",
        "--eval-every", "100",
        "--eval-tokens", "262144",
        "--core-metric-every", "-1",
        "--sample-every", "100",
        "--save-every", "-1",
        "--num-iterations", "$TrainIterations",
        "--run", "$Run",
        "--model-tag", "$ModelTag"
    )
}

if ($requestedStages.Contains("eval")) {
    Write-Step "Running a quick single-GPU evaluation of the base checkpoint"
    Run-Python @(
        "-m", "scripts.base_eval",
        "--device-type", "cuda",
        "--model-tag", "$ModelTag",
        "--device-batch-size", "1",
        "--max-per-task", "50",
        "--split-tokens", "131072"
    )
}

if ($requestedStages.Contains("sft")) {
    Ensure-IdentityConversations
    Write-Step "Running supervised fine-tuning from the base checkpoint"
    Run-Python @(
        "-m", "scripts.chat_sft",
        "--device-type", "cuda",
        "--model-tag", "$ModelTag",
        "--max-seq-len", "$MaxSeqLen",
        "--device-batch-size", "$DeviceBatchSize",
        "--total-batch-size", "$TotalBatchSize",
        "--eval-every", "200",
        "--eval-tokens", "262144",
        "--chatcore-every", "-1",
        "--num-iterations", "$SftIterations",
        "--run", "$Run"
    )
}

if ($requestedStages.Contains("chat-cli")) {
    Write-Step "Launching CLI chat against the SFT checkpoint"
    Run-Python @(
        "-m", "scripts.chat_cli",
        "--device-type", "cuda",
        "--source", "sft",
        "--model-tag", "$ModelTag",
        "--prompt", "$Prompt"
    )
}

if ($requestedStages.Contains("chat-web")) {
    Write-Step "Launching the local web UI"
    Run-Python @(
        "-m", "scripts.chat_web",
        "--device-type", "cuda",
        "--source", "sft",
        "--model-tag", "$ModelTag",
        "--host", "127.0.0.1",
        "--port", "$Port"
    )
}