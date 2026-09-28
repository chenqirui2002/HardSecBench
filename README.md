# HardSecBench

This is the official repository for **HardSecBench: Benchmarking the Security Awareness of LLMs for Hardware Code Generation**, accepted at **IJCAI 2026**.

[![arXiv](https://img.shields.io/badge/arXiv-2601.13864-b31b1b.svg)](https://arxiv.org/abs/2601.13864)

## Overview

HardSecBench is a comprehensive benchmark for evaluating Large Language Models (LLMs) on hardware security code generation tasks, covering both Verilog and C implementations. It consists of two main components:

1. **Evaluation System** - Comprehensive evaluation framework for testing LLMs on hardware security tasks
2. **Data Generation System** - Automated generation of hardware security test cases using multi-agent architecture

## Quick Start

### Prerequisites

- Python 3.8+
- Conda (recommended)
- Icarus Verilog (for Verilog test cases)
- GCC (for C test cases)

### System Dependencies

**Install Icarus Verilog:**

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install iverilog

# macOS
brew install icarus-verilog

# Verify installation
iverilog -v
```

**Install GCC:**

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential

# macOS (Install Xcode Command Line Tools)
xcode-select --install

# Verify installation
gcc --version
```

### Python Environment Setup

1. **Clone the repository**
```bash
git clone https://github.com/chenqirui2002/HardSecBench.git
cd HardSecBench
```

2. **Create and activate conda environment**
```bash
conda create -n hardsecbench python=3.12
conda activate hardsecbench
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Extract benchmark data**
```bash
# Extract the compressed benchmark data
cd data
unzip HardSecBench_Data.zip
# or if it's a tar.gz file:
# tar -xzf HardSecBench_Data.tar.gz
cd ..
```

5. **Configure API keys**

Copy the example environment files and fill in your API keys:

```bash
# For evaluation
cp eval/.env.example eval/.env
# Edit eval/.env and add your API keys

# For data generation
cp data_generation/.env.example data_generation/.env
# Edit data_generation/.env and add your API keys
```

## Evaluation

Evaluate LLMs on the HardSecBench benchmark.

### Standard Evaluation (Pass@1)

```bash
cd eval

# Run evaluation
bash eval_all.sh
```

**Configuration** - Edit `eval/eval_all.sh`:
```bash
MODEL="gpt-4"                           # Model to evaluate
DATA_PATH="../data/HardSecBench_Data"   # Use provided benchmark
# Or use your own generated data:
# DATA_PATH="../data_generation/outputs/batchall_YYYYMMDD_HHMMSS_MODEL"

TEMPERATURE=0.6
MAX_ITERATIONS=5                 # Max refinement iterations
MAX_ATTEMPTS=3                   # Max attempts per iteration
WORKERS=20                       # Parallel workers
SECURITY_HINT="0"                # 0=none, 1=general, 2=cwe-specific
```

**Using Your Own Generated Data:**
1. Generate data using `data_generation/batchall.sh`
2. Note the output directory path (e.g., `data_generation/outputs/batchall_20260108_120000_gpt-4`)
3. Edit `eval/eval_all.sh` and set `DATA_PATH` to your generated data directory
4. Run `bash eval_all.sh`

### Pass@k Evaluation

For Pass@k evaluation (multiple independent runs per test case), edit `eval/eval_all.sh`:

```bash
PASS_AT_K=5                      # Number of independent runs
GEN_WORKERS=30                   # Workers for code generation
EVAL_WORKERS=30                  # Workers for evaluation
```

Then run:
```bash
bash eval_all.sh
```

### Local Transformers Models

To evaluate local models (e.g., RTLCoder, VeriThoughts):

```bash
cd eval

# Edit eval_transformers.sh to configure model and GPU
bash eval_transformers.sh
```

**Configuration** - Edit `eval/eval_transformers.sh`:
```bash
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
CUDA_VISIBLE_DEVICES="0"
BATCH_SIZE=1
LANGUAGE_FILTER="verilog"        # or "c" or leave empty
SECURITY_HINT="0,1,2"            # Evaluate multiple hint levels
```

### Evaluation Metrics

The evaluation system computes:

- **Functional Correctness** - Pass rate on functional requirements
- **Security Correctness** - Pass rate on security requirements
- **Overall Pass Rate** - Combined functional and security correctness
- **Pass@k** - Probability of at least one correct solution in k attempts
- **Iteration Statistics** - Number of refinement iterations needed

Results are saved in `eval/outputs/` with detailed logs and per-test-case breakdowns.

### Security Hints

The evaluation system supports three levels of security hints:

- **Level 0 (None)**: No security information provided
- **Level 1 (General)**: General security awareness prompt
- **Level 2 (CWE-specific)**: Specific CWE information provided

Configure via `SECURITY_HINT` parameter in evaluation scripts.

## Data Generation

Generate new test cases using the multi-agent data generation system.

### Step 1: Generate Test Cases

```bash
cd data_generation

# Run batch generation
bash batchall.sh
```

**Configuration** - Edit `data_generation/batchall.sh`:
- `MODEL` - LLM model to use
- `CWE_ID` - Target CWE category (or leave empty for all)
- `NUM_SEEDS` - Number of test cases to generate
- `TEMPERATURE` - Model temperature
- `MAX_ITERATIONS` - Maximum refinement iterations

Also configure `data_generation/.env` for API keys. See `.env.example` for details.

**Output**: Generated data will be saved to `data_generation/outputs/batchall_YYYYMMDD_HHMMSS_MODEL/`

### Step 2: Filter High-Quality Test Cases

Filter generated test cases based on testbench quality:

```bash
cd data_generation

# Edit filter.sh to configure DATA_PATH
bash filter.sh
```

**Configuration** - Edit `data_generation/filter.sh`:
```bash
# Set DATA_PATH to your generated data directory
DATA_PATH="outputs/batchall_YYYYMMDD_HHMMSS_MODEL"
MUTATIONS=5                    # Number of mutations to test
MUTATION_THRESHOLD=0.5         # Min mutation detection rate
COVERAGE_THRESHOLD=0.8         # Min code coverage required
```

**Example**: If you generated data and got output directory `outputs/batchall_20260108_120000_gpt-4`, set:
```bash
DATA_PATH="outputs/batchall_20260108_120000_gpt-4"
```

This script uses mutation testing to verify that testbenches can effectively detect bugs.

**Output**: Filtering creates `result_filtered.json` in the data directory, containing only high-quality test cases.

### Step 3: Use Generated Data for Evaluation

After generation (and optional filtering), use your generated data for evaluation:

```bash
cd eval

# Edit eval_all.sh and set DATA_PATH to your generated data
# DATA_PATH="../data_generation/outputs/batchall_YYYYMMDD_HHMMSS_MODEL"
bash eval_all.sh
```

**Data Loading Priority**: The evaluation system automatically loads data in this order:
1. `result_filtered.json` (if exists - filtered high-quality test cases)
2. `result.json` (if result_filtered.json doesn't exist - all generated test cases)

This means if you ran the filter script, evaluation will automatically use only the filtered test cases.

## Benchmark Data

The benchmark data is located in `data/HardSecBench_Data/` (after extraction) and contains:

- **Test cases** covering multiple CWE (Common Weakness Enumeration) categories
- **Verilog and C implementations** for hardware security vulnerabilities
- **Functional and security requirements** for each test case
- **Reference implementations** and test benches

### Data Structure

```
data/HardSecBench_Data/
├── CWE-XXXX/                    # CWE category
│   ├── seed_id_1/               # Individual test case
│   │   ├── problem.json         # Problem description
│   │   ├── topmodule.v          # Reference Verilog implementation
│   │   ├── module.c             # Reference C implementation
│   │   ├── tb_*.v               # Verilog testbenches
│   │   └── tb_*.c               # C testbenches
│   └── seed_id_2/
│       └── ...
└── result.json                  # Benchmark metadata
```

## Advanced Features

### Code Deduplication

For Pass@k evaluation, enable code deduplication to avoid redundant testing:

```bash
DEDUPLICATE_CODE=true
```

This groups identical code (ignoring whitespace/comments) and only evaluates unique implementations.

### Parallel Execution

Control parallelism with worker configuration:

```bash
WORKERS=20              # For Pass@1 evaluation
GEN_WORKERS=30          # For Pass@k code generation
EVAL_WORKERS=30         # For Pass@k evaluation
```

### LangSmith Tracing

Enable LangSmith for detailed LLM call tracing:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=HardSecBench
```

## API Configuration


### Environment Variables

**Evaluation** (`eval/.env`):
```bash
# Target LLM (model being evaluated)
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# Collaborator Agent (helper for code refinement)
COLLABORATOR_MODEL=gpt-4
COLLABORATOR_API_KEY=your_api_key
COLLABORATOR_BASE_URL=https://api.openai.com/v1
```

**Data Generation** (`data_generation/.env`):
```bash
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional
```

**Note**: You can use different API keys/models for each agent, or use the same key for all.


## Citation

If you find HardSecBench useful, please cite our paper:

```bibtex
@inproceedings{ijcai2026p57,
  title     = {HardSecBench: Benchmarking the Security Awareness of LLMs for Hardware Code Generation},
  author    = {Chen, Qirui and Shuai, Jingxian and Chen, Shuangwu and Ye, Shenghao and Wen, Zijian and Su, Xufei and Jin, Jie and Li, Jiangming and Chen, Jun and Tan, Xiaobin and Yang, Jian},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on
               Artificial Intelligence, {IJCAI-26}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Diego Calvanese},
  pages     = {500--508},
  year      = {2026},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2026/57},
  url       = {https://doi.org/10.24963/ijcai.2026/57},
}
```
