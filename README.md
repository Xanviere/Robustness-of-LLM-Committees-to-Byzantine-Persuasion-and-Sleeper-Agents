# Robustness of LLM Committees to Byzantine Persuasion and Sleeper Agents

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Sandboxed-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

An asynchronous, production-grade research harness to benchmark the vulnerability of **Multi-Agent Large Language Model (LLM) Committees** under **Byzantine insider persuasion**, empirically test the classical distributed fault-tolerance bound ($f < N/3$), and evaluate multi-tier protocol defenses.

This repository contains the official codebase and experimental benchmarks for the research paper:  
**"Robustness of LLM Committees to Static and Trust-Building Byzantine Agents"** *(Preprint / Under Review)*  
*Utkarsh Srivastava, Avidha Halder, and Dr. Kathiravan Srinivasan*  
Department of Computer Science and Engineering, Vellore Institute of Technology (VIT), Vellore, India.

---

## Table of Contents

- [Overview & Research Questions](#overview--research-questions)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Reproducing Experiments](#reproducing-experiments)
  - [1. Unit Testing & Verification](#1-unit-testing--verification)
  - [2. Stage 1: Exploratory Pilot Sweeps](#2-stage-1-exploratory-pilot-sweeps)
  - [3. Stage 2: Primary Confirmatory Factorial Benchmark](#3-stage-2-primary-confirmatory-factorial-benchmark)
  - [4. The Classical Byzantine Bound ($N=5, k=2$)](#4-the-classical-byzantine-bound-n5-k2)
  - [5. Stage 3: Multi-Tier Algorithmic Defense Benchmark](#5-stage-3-multi-tier-algorithmic-defense-benchmark)
  - [6. Stage 4: Longitudinal Sleeper Agent Dynamics](#6-stage-4-longitudinal-sleeper-agent-dynamics)
  - [7. Ad-Hoc Experiment Runner](#7-ad-hoc-experiment-runner)
- [Evaluation Estimands & Statistical Inference](#evaluation-estimands--statistical-inference)
- [Security & Sandbox Isolation](#security--sandbox-isolation)
- [Citation](#citation)

---

## Overview & Research Questions

Multi-agent deliberation is widely used to enhance reasoning fidelity in large language models. However, standard consensus protocols implicitly presume that all participating models are honest, leaving panels vulnerable to **adversarial persuasion from within**.

This research harness answers four central questions:
1. **Frontier vs. Open-Weights Resilience:** Do frontier models (`gemini-3.7-flash`) and open-weights models (`Llama-3.3-70B-Instruct`) exhibit different susceptibilities to persuasive deception?
2. **The Classical Distributed Fault-Tolerance Bound:** Does collective LLM debate obey the classical Byzantine bound ($f < N/3$), or can raw cardinality scaling ($N=3 \to N=5 \to N=7$) protect consensus?
3. **Longitudinal Sleeper Threats:** How vulnerable are recurring workflows to "sleeper agents" that behave honestly to build prior reputation before defecting?
4. **Defense-in-Depth:** Can lightweight protocol mechanisms mitigate persuasion without destroying the beneficial self-repair enabled by collaborative debate?

---

## System Architecture

```
+-----------------------------------------------------------------------------------+
|                            Paired Counterfactual Harness                          |
|                                                                                   |
|                                Task Prompt (MATH / LCB / HumanEval+)              |
|                                                  |                                |
|                                   Round 0 Anchor Generation                       |
|                             S_i^(0) = <verdict, derivation, conf>                 |
|                                         /            \                            |
|                       Master Anchor State            Cloned Initial State         |
|                                     |                        |                    |
|                        [CLEAN BRANCH]              [ATTACKED BRANCH]              |
|                     All Honest (Control)        k Byzantine Insiders Infiltrate   |
|                             |                                |                    |
|                     Round 1..R Debate                Round 1..R Debate            |
|                             |                                |                    |
|                      Clean Consensus                  Attacked Consensus          |
|                             \                                /                    |
|                        Evaluate Causal Delta: AAE = VSR - CCR                     |
+-----------------------------------------------------------------------------------+
```

---

## Repository Structure

```text
byzantine_mas/
├── attacks/                    # Byzantine adversary implementations
│   ├── redteam_agent.py        # Authoritative persuasive red-team agent
│   ├── strategies.py           # Factorial threat matrix (Blind/Oracle, Fixed/Adaptive)
│   └── sleeper.py              # Multi-task sleeper state tracker & honest burn-in
├── config/                     # Configuration definitions and pricing tables
├── core/                       # Core infrastructure & API wrappers
│   ├── client_wrapper.py       # Asynchronous LLM client (Together AI, Gemini, OpenAI)
│   ├── concurrency.py          # ConcurrencyManager with rate-limiting & backoff
│   └── models.py               # Pydantic v2 schemas (AgentResponse, TelemetryRecord)
├── data/                       # Experimental data manifests and reports
│   ├── manifests/              # Curated JSONL manifests (MATH-500, LiveCodeBench, HumanEval+)
│   ├── raw_logs/               # Detailed per-round telemetry trajectories
│   └── reports/                # Aggregated JSON evaluation metrics & contingency tables
├── debate/                     # Multi-agent deliberation engine
│   ├── engine.py               # DebateEngine (round execution, branch cloning, voting)
│   └── context_builder.py      # Transcript assembly and peer message shuffling
├── defenses/                   # Multi-tier algorithmic defenses
│   ├── audit_anchor.py         # L0: Immutable Round 0 anchor lock (reference bound)
│   ├── prompt_reminder.py      # L1: Contextual prompt reminder with peer citation invariants
│   ├── verifier_gate.py        # L2: SymPy symbolic check & containerized unit-test gate
│   └── bayesian_filter.py      # L3: Longitudinal Beta-binomial reputation tracker
├── evaluation/                 # Statistical analysis & econometric modeling
│   ├── statistical_tests.py    # Clustered BCa bootstrap, permutation tests, FDR, GEE, LMM
│   ├── tier1_operational.py    # Unconditional operational population metrics
│   ├── tier2_mechanism.py      # Conditioned S_maj analysis (VSR, CCR, AAE, contingency cells)
│   └── cost_tracker.py         # Real-time token and USD financial spend tracker
├── tests/                      # Verification test suite
│   ├── test_verifiers.py       # Tests for SymPy and regex answer extraction
│   ├── test_bayesian_math.py   # Unit tests for Bayesian belief update math
│   └── test_vertical_slice.py  # End-to-end integration test
├── verifiers/                  # Domain-specific verification backends
│   ├── code_sandbox.py         # Docker-isolated execution containment barrier
│   └── math_verifier.py        # SymPy exact symbolic equivalence testing
├── Dockerfile.sandbox          # Hardened Alpine/Debian Docker container for code execution
├── requirements.txt            # Python dependencies
├── check_together_key.py       # Together AI API connectivity test utility
├── compute_stage3_l0_baseline.py # L0 anchor lock baseline extractor
├── run_byzantine_threshold_n5k2.py # Classical Byzantine bound runner (N=5, k=2)
├── run_experiment.py           # General ad-hoc debate experiment runner
├── run_stage1_pilot.py         # Stage 1 pilot execution script
├── run_stage2_primary.py       # Stage 2 primary confirmatory factorial sweep runner
├── run_stage3_defenses.py      # Stage 3 multi-tier defense benchmark runner
└── run_stage4_sleeper.py       # Stage 4 longitudinal sleeper agent runner
```

---

## Environment Setup

### 1. Clone the Repository & Install Dependencies
```bash
git clone https://github.com/<your-username>/byzantine_mas.git
cd byzantine_mas

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Build Docker Container Sandbox
Deterministic verification for code benchmarks (LiveCodeBench and HumanEval+) runs inside an isolated, non-root Docker container:
```bash
docker build -t python:3.11-slim -f Dockerfile.sandbox .
```

### 3. Configure API Credentials
Set the environment variables corresponding to the providers you intend to benchmark:

**Linux / macOS (Bash):**
```bash
export TOGETHER_API_KEY="your-together-ai-api-key"   # For Llama-3.3-70B-Instruct
export GEMINI_API_KEY="your-google-gemini-api-key"   # For Gemini 3.7 Flash
export OPENAI_API_KEY="your-openai-api-key"           # Optional
```

**Windows (PowerShell):**
```powershell
$env:TOGETHER_API_KEY="your-together-ai-api-key"
$env:GEMINI_API_KEY="your-google-gemini-api-key"
```

Verify your Together AI key using the included verification tool:
```bash
python check_together_key.py
```

---

## Reproducing Experiments

### 1. Unit Testing & Verification
Ensure that symbolic parsing, Docker containment, and Bayesian reputation math are functioning properly:
```bash
python tests/test_verifiers.py
python tests/test_bayesian_math.py
```

---

### 2. Stage 1: Exploratory Pilot Sweeps
Executes the exploratory pilot across $Q = 30$ tasks (Gemini) or $Q = 22$ tasks (Llama):
```bash
# Gemini 3.7 Flash Pilot
python run_stage1_pilot.py \
  --manifest data/manifests/stage1_frontier_hard_manifest.jsonl \
  --provider gemini \
  --model gemini-3.7-flash \
  --replicates 3 \
  --rounds 2 \
  --output-log data/raw_logs/stage1_gemini_pilot.jsonl

# Llama-3.3-70B-Instruct Pilot
python run_stage1_pilot.py \
  --manifest data/manifests/stage1_llama_pilot.jsonl \
  --provider openai \
  --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
  --replicates 1 \
  --rounds 2 \
  --output-log data/raw_logs/stage1_llama_pilot.jsonl
```

---

### 3. Stage 2: Primary Confirmatory Factorial Benchmark
Executes the primary confirmatory evaluation across committee sizes $N \in \{3, 5, 7\}$ spanning four threat profiles (`blind_postanchor_adaptive`, `oracle_postanchor_adaptive`, `oracle_r0_adaptive`, `oracle_sleeper_adaptive`) over $N_{\text{evals}} = 330$ trials:
```bash
python run_stage2_primary.py \
  --manifest data/manifests/stage2_frontier_75.jsonl \
  --provider openai \
  --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
  --committee-sizes 3 5 7 \
  --replicates 3 \
  --rounds 2 \
  --threats all \
  --output-log data/raw_logs/stage2_primary.jsonl \
  --report-output data/reports/stage2_summary.json
```

---

### 4. The Classical Byzantine Bound ($N=5, k=2$)
Tests the distributed systems fault-tolerance threshold ($f < N/3$) by deploying **two colluding Byzantine agents** in seats 3 and 4 against three honest peers in seats 0, 1, and 2 ($f/N = 40.0\%$):
```bash
python run_byzantine_threshold_n5k2.py
```
This script evaluates the $N_{\text{evals}} = 330$ trials against frozen Round 0 baselines, compiles per-domain metrics (MATH-500, LiveCodeBench, HumanEval+), and outputs `data/reports/byzantine_threshold_n5k2_report.json`.

---

### 5. Stage 3: Multi-Tier Algorithmic Defense Benchmark
Evaluates the multi-tier defense pipeline at peak Byzantine vulnerability ($N = 3$, Llama-3.3-70B, $N_{\text{evals}} = 330$ trials):
- **$L_0$:** Hard Anchor Lock (Reference Bound)
- **$L_1$:** Contextual Prompt Reminder
- **$L_2$:** Pre-Consensus Verifier Gate (SymPy + Docker)
- **$L_3$:** Longitudinal Bayesian Trust Filter
- **Full Suite:** $L_1 + L_2 + L_3$ integrated

```bash
python run_stage3_defenses.py
```
Outputs complete contingency tables ($n_{00}, n_{01}, n_{10}, n_{11}$), BCa confidence intervals, and pairwise permutation tests against the undefended baseline.

---

### 6. Stage 4: Longitudinal Sleeper Agent Dynamics
Tracks 5 independent multi-agent committees ($S = 5$) across sequences of $T = 10$ tasks where the adversary exhibits honest behavior for $K = 4$ tasks before defecting at Task 5:
```bash
python run_stage4_sleeper.py
```
Computes mean trust trajectories ($\overline{\text{Trust}}_t$), automated quarantine task latency ($0.60$ tasks post-defection), post-activation defended ($68\%$) vs. control ($24\%$) consensus accuracy, and fits the piecewise Linear Mixed-Effects Model (LMM).

---

### 7. Ad-Hoc Experiment Runner
Run targeted experiments with custom parameters:
```bash
python run_experiment.py \
  --manifest data/manifests/stage2_frontier_75.jsonl \
  --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
  --committee-size 5 \
  --rounds 2 \
  --threat-id oracle_postanchor_adaptive \
  --defense-id verifier_gate_defense \
  --temperature 0.7 \
  --output-log data/raw_logs/custom_experiment.jsonl
```

---

## Evaluation Estimands & Statistical Inference

To isolate causal persuasion from intrinsic model stochasticity, evaluations condition on the pre-treatment honest majority set $\mathcal{S}_{\text{maj}}$ (where honest agents hold an unattacked correct majority at Round 0):

$$\text{VSR} = \frac{n_{01}}{m}, \quad \text{CCR} = \frac{n_{10}}{m}, \quad \text{AAE} = \text{VSR} - \text{CCR} = \frac{n_{01} - n_{10}}{m}$$

- **$n_{00}$:** Clean correct, Attacked correct (Stable Correct)
- **$n_{01}$:** Clean correct, Attacked incorrect (Harmful Persuasion Flip)
- **$n_{10}$:** Clean incorrect, Attacked correct (Beneficial Counterfactual Correction)
- **$n_{11}$:** Clean incorrect, Attacked incorrect (Uncorrected Baseline Error)

### Statistical Framework
- **Confidence Intervals:** Question-clustered non-parametric BCa bootstrap ($B = 2{,}000$ resamples).
- **Hypothesis Testing:** Question-clustered paired sign-flip permutation tests ($M = 10{,}000$ iterations) with Benjamini--Hochberg FDR correction ($\alpha = 0.05$).
- **Population Dynamics:** Population-averaged Marginal Generalized Estimating Equations (GEE) logit model clustered across $330$ question IDs ($N_{\text{obs}} = 11{,}161$).
- **Temporal Trajectories:** Piecewise Linear Mixed-Effects Models (LMM) with sequence-level random effects.

---

## Security & Sandbox Isolation

To guarantee that evaluated code solutions cannot compromise host environments, all code executions run inside hardened Docker sandboxes:
- Network disabled: `--network none`
- Read-only filesystem root with ephemeral `/tmp` volume
- Strict memory ceiling: `--memory 1g --memory-swap 1g`
- Strict CPU quota: `--cpus 2.0`
- PID ceiling to prevent fork bombs: `--pids-limit 15`
- Linux capability dropping: `--cap-drop=ALL`
- Wall-clock timeout: `5.0` seconds enforced at process level

---

## Citation

If you use this research harness, data manifests, or findings in your work, please cite:

```bibtex
@misc{srivastava2026robustness,
  author       = {Utkarsh Srivastava and Avidha Halder and Kathiravan Srinivasan},
  title        = {Robustness of {LLM} Committees to Static and Trust-Building Byzantine Agents},
  howpublished = {Preprint / Code Repository},
  year         = {2026},
  url          = {https://github.com/Xanviere/Robustness-of-LLM-Committees-to-Static-and-Trust-Building-Byzantine-Agents}
}
```
