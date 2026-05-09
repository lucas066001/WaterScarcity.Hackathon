# WaterScarcity.Hackathon

## Overview

This repository contains the codebase developed for the **Capgemini Invent 2025 Water Scarcity Hackathon**. The project is organized into two complementary phases:

- **Phase 1 — Streamflow forecasting:** probabilistic prediction of weekly river discharge using spatio-temporal, meteorological, hydrological, and geospatial data from France and Brazil.
- **Phase 2 — Water-allocation policy design:** simulation and optimization of water-management policies using game-theoretic reasoning and evolutionary search.

The repository supports reproducible execution of the main workflows, from data preprocessing and feature engineering to model training, evaluation, policy simulation, and optimization.

---

## Table of Contents

- [WaterScarcity.Hackathon](#waterscarcityhackathon)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Objectives](#objectives)
    - [Phase 1: Streamflow Forecasting](#phase-1-streamflow-forecasting)
    - [Phase 2: Water-Allocation Policy Design](#phase-2-water-allocation-policy-design)
  - [Setup and Installation](#setup-and-installation)
    - [Recommended Environment Strategy](#recommended-environment-strategy)
    - [Phase 1 Environment](#phase-1-environment)
    - [Phase 2 Environment](#phase-2-environment)
  - [Usage](#usage)
    - [Running Phase 1](#running-phase-1)
    - [Running Phase 2](#running-phase-2)
  - [Reproducibility Notes](#reproducibility-notes)
  - [Data Availability](#data-availability)
  - [Acknowledgements](#acknowledgements)

---

## Project Structure

```text
.
├── pyproject.toml
├── README.md
├── phase_one_waterflow_prediction_machine_learning/
│   ├── data/
│   │   ├── evaluation/
│   │   └── ...
│   ├── dataset/
│   │   ├── Brazil/
│   │   ├── France/
│   │   └── ...
│   ├── dataset_mini_challenge/
│   ├── models/
│   ├── src/
│   │   ├── notebooks/
│   │   └── ...
│   ├── requirements.txt
│   └── README.md
└── phase_two_game_theroy_genetic_optimization/
    ├── src/
    │   ├── policies/
    │   └── ...
    ├── a*.ipynb
    ├── b*.ipynb
    ├── c.multi_scenarios_final.ipynb
    ├── requirements.txt
    └── README.md
```

> Note: the Phase 2 folder name preserves the original hackathon repository naming.

---

## Objectives

### Phase 1: Streamflow Forecasting

The goal of Phase 1 is to develop robust probabilistic models for weekly river-flow prediction across hydrometric stations in France and Brazil.

The forecasting workflow includes:

- data cleaning and preprocessing;
- hydrology-guided feature engineering;
- spatial and temporal validation splits;
- model training and hyperparameter tuning;
- probabilistic prediction with quantile-based uncertainty estimates;
- evaluation of predictive accuracy and interval calibration.

Main modeling components include:

- Quantile Random Forests;
- XGBoost-based models;
- XGB-QRF hybrid ensemble;
- LightGBM;
- Explainable Boosting Machines;
- MAPIE-based uncertainty estimation;
- SHAP-based feature-importance analysis.

Feature engineering includes:

- principal component analysis of correlated variables;
- lagged streamflow features;
- moving-average streamflow features;
- cyclical temporal encodings;
- snow-index construction;
- spatial and hydrological clustering.

### Phase 2: Water-Allocation Policy Design

The goal of Phase 2 is to design and optimize adaptive water-management policies in a multi-agent simulation environment.

The policy-design workflow includes:

- simulation of heterogeneous water users;
- definition of priority-aware water quotas;
- design of incentives through fines and subsidies;
- testing of baseline, high-cooperation, and optimized scenarios;
- evolutionary optimization of interpretable policy parameters;
- robustness analysis under forecast bias and uncertainty.

The final policy framework combines:

- quota-based allocation;
- priority scaling;
- crisis-level rules;
- subsidy and penalty mechanisms;
- evolutionary search over policy parameters.

---

## Setup and Installation

### Recommended Environment Strategy

Phase 1 and Phase 2 rely on different dependency stacks. To avoid dependency conflicts, we recommend using **separate virtual environments** for each phase.

The repository includes a root-level `pyproject.toml` with separate optional dependency groups:

- `phase1`
- `phase2`

Use the corresponding group depending on which part of the project you want to run.

Recommended Python version:

```text
Python 3.10
```

Python 3.10 is recommended because it provides broad compatibility with the geospatial, machine-learning, and notebook dependencies used in the project.

---

### Phase 1 Environment

From the repository root:

```bash
python3.10 -m venv .venv-phase1
source .venv-phase1/bin/activate
python -m pip install --upgrade pip
pip install -e ".[phase1]"
```

On Windows PowerShell:

```powershell
py -3.10 -m venv .venv-phase1
.venv-phase1\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[phase1]"
```

To deactivate the environment:

```bash
deactivate
```

---

### Phase 2 Environment

From the repository root:

```bash
python3.10 -m venv .venv-phase2
source .venv-phase2/bin/activate
python -m pip install --upgrade pip
pip install -e ".[phase2]"
```

On Windows PowerShell:

```powershell
py -3.10 -m venv .venv-phase2
.venv-phase2\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[phase2]"
```

To deactivate the environment:

```bash
deactivate
```

---

## Usage

### Running Phase 1

Phase 1 workflows are located in:

```text
phase_one_waterflow_prediction_machine_learning/
```

The main notebooks are located under:

```text
phase_one_waterflow_prediction_machine_learning/src/notebooks/
```

Recommended workflow:

1. Create and activate the Phase 1 environment.
2. Ensure the required dataset folders are available.
3. Open the notebooks in `src/notebooks/`.
4. Start from the summary or pipeline notebook described in the Phase 1 README.
5. Run preprocessing, feature engineering, model training, and evaluation notebooks in order.

Useful locations:

```text
phase_one_waterflow_prediction_machine_learning/data/
phase_one_waterflow_prediction_machine_learning/models/
phase_one_waterflow_prediction_machine_learning/src/notebooks/
```

For detailed execution order, see:

```text
phase_one_waterflow_prediction_machine_learning/README.md
```

---

### Running Phase 2

Phase 2 workflows are located in:

```text
phase_two_game_theroy_genetic_optimization/
```

The main notebooks are organized by scenario type:

```text
a*.ipynb                         # Single-scenario analyses
b*.ipynb                         # Multi-scenario analyses
c.multi_scenarios_final.ipynb    # Final multi-scenario results
```

Recommended workflow:

1. Create and activate the Phase 2 environment.
2. Open the Phase 2 notebooks.
3. Start with the single-scenario notebooks prefixed with `a`.
4. Continue with the multi-scenario notebooks prefixed with `b`.
5. Use `c.multi_scenarios_final.ipynb` for the final multi-scenario evaluation.

Policy functions are located in:

```text
phase_two_game_theroy_genetic_optimization/src/policies/
```

The final custom policy implementation is located in:

```text
phase_two_game_theroy_genetic_optimization/src/policies/custom_final_policies.py
```

For detailed scenario and policy descriptions, see:

```text
phase_two_game_theroy_genetic_optimization/README.md
```

---

## Reproducibility Notes

This repository separates the environments for Phase 1 and Phase 2 because the two workflows use different versions of several scientific Python packages.

For example, the two phases use different dependency versions for packages such as:

```text
scikit-learn
scipy
plotly
Pygments
```

Installing both phases in the same environment is therefore not recommended unless the combined environment has been explicitly tested.

Recommended reproducibility practice:

```text
Use .venv-phase1 for Phase 1.
Use .venv-phase2 for Phase 2.
Do not mix both environments unless necessary.
```

The repository includes:

- a root-level `pyproject.toml`;
- phase-specific optional dependency groups;
- phase-specific code and notebooks;
- precomputed outputs where applicable;
- phase-specific README files with additional details.

---

## Data Availability

The data used in this project were provided by the organizers of the Capgemini Invent 2025 Water Scarcity Hackathon.

The repository is structured to work with the original dataset layout used during the hackathon. Some raw data files may not be redistributed directly in this repository depending on the licensing terms of the original dataset.

Expected Phase 1 data locations include:

```text
phase_one_waterflow_prediction_machine_learning/dataset/
phase_one_waterflow_prediction_machine_learning/dataset_mini_challenge/
phase_one_waterflow_prediction_machine_learning/data/
```

Generated outputs are saved under:

```text
phase_one_waterflow_prediction_machine_learning/models/
phase_one_waterflow_prediction_machine_learning/data/evaluation/
```

---

## Acknowledgements

We thank the organizers of the **Capgemini Invent 2025 Water Scarcity Hackathon** for providing the challenge framework, datasets, and evaluation platform.

We also acknowledge the open-source Python ecosystem and the contributors of the scientific, geospatial, machine-learning, visualization, and notebook libraries used in this project.
