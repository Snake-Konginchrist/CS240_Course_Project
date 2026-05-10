# CS240 Course Project

This repository contains the course project materials for **CS240: Algorithm
Design and Analysis, Spring 2026**.

## Project Topic

**Scalable Graph Partitioning for Communication-Efficient Distributed
Optimization**

The project is based on **Topic 2: Community Detection / Graph Partitioning**.
The application is adapted to distributed optimization: graph partitions are
used to reduce cross-worker communication while keeping computational load
balanced.

## Implemented Methods

- Greedy balanced partitioning baseline
- Recursive spectral bisection
- Kernighan--Lin pairwise refinement for k-way partitioning
- Consensus averaging simulation for cross-partition communication analysis

## Project Structure

| Path | Description |
| --- | --- |
| `src/graphs.py` | Graph generation and benchmark graph loading |
| `src/algorithms.py` | Greedy, spectral, and Kernighan--Lin partitioning |
| `src/metrics.py` | Cut weight, normalized cut, and balance metrics |
| `src/simulation.py` | Consensus averaging simulation |
| `src/experiments.py` | End-to-end experiment runner |
| `src/plotting.py` | Figure generation |
| `tests/` | Lightweight unit tests |
| `outputs/` | Generated CSV results and figures |
| `Proposal.tex`, `Proposal.pdf` | English project proposal |
| `Proposal.zh.tex`, `Proposal.zh.pdf` | Chinese project proposal |
| `Report.tex`, `Report.pdf` | English experimental report |
| `Report.zh.tex`, `Report.zh.pdf` | Chinese experimental report |

## Setup

Create and activate the virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies and the local package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run Experiments

```bash
python src/experiments.py --output-dir outputs --seed 7
```

This writes:

- `outputs/partition_results.csv`
- `outputs/figures/cut_weight.png`
- `outputs/figures/runtime_seconds.png`
- `outputs/figures/balance_ratio.png`
- `outputs/figures/consensus_error.png`

## Run Tests

```bash
python -m pytest
```

## Build Documents

```bash
pdflatex Proposal.tex
xelatex Proposal.zh.tex
pdflatex Report.tex
xelatex Report.zh.tex
```

Run each command twice if PDF outlines need to be refreshed.

## Results Snapshot

The current run shows that Kernighan--Lin refinement achieves the lowest
average cut weight and cross-partition communication among the implemented
methods. Spectral partitioning is also strong on structured grid and geometric
graphs.

<table>
  <tr>
    <td><img src="outputs/figures/cut_weight.png" alt="Cut weight comparison"></td>
    <td><img src="outputs/figures/runtime_seconds.png" alt="Runtime comparison"></td>
    <td><img src="outputs/figures/balance_ratio.png" alt="Balance ratio comparison"></td>
  </tr>
  <tr>
    <td align="center">Cut weight</td>
    <td align="center">Runtime</td>
    <td align="center">Balance ratio</td>
  </tr>
</table>

## Known Deviations

The original Kernighan--Lin paper focuses on graph bisection. This project uses
pairwise Kernighan--Lin bisection as a refinement step for k-way partitioning,
initialized by spectral partitioning. The consensus simulation keeps the
underlying graph fixed and uses partitions to measure cross-worker traffic, so
partitioning affects communication placement rather than the consensus dynamics
itself.
