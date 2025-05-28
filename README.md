# WaterScarcity.Hackathon

## Overview

This repository contains the full codebase, data processing, modeling, and optimization workflows developed for the Capgemini 2025 Water Scarcity Hackathon. The project is organized in two main phases:

- **Phase One:** Machine learning-based prediction of river water flow using spatio-temporal data from Brazil and France.
- **Phase Two:** Game theory and genetic optimization for water management policy design and simulation.

The repository is structured to enable full reproducibility of results, from raw data preprocessing to advanced model training, evaluation, and policy optimization.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Objectives](#objectives)
- [Phase One: Water Flow Prediction](#phase-one-water-flow-prediction)
  - [Data Preprocessing & Feature Engineering](#data-preprocessing--feature-engineering)
  - [Model Training & Evaluation](#model-training--evaluation)
  - [Reproducibility](#reproducibility)
- [Phase Two: Game Theory & Optimization](#phase-two-game-theory--optimization)
  - [Simulation Scenarios](#simulation-scenarios)
  - [Policy Optimization](#policy-optimization)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Contributors](#contributors)
- [License](#license)

---

## Project Structure
. ├── phase_one_waterflow_prediction_machine_learning/ 
  │ ├── data/ # Input, evaluation, and intermediate data 
  │ ├── dataset/ # Raw datasets (Brazil, France) 
  │ ├── dataset_mini_challenge/ 
  │ ├── models/ # Saved models and results 
  │ ├── src/ # Source code and notebooks 
  │ ├── requirements.txt 
  │ └── README.md 
  ├── phase_two_game_theroy_genetic_optimization/ 
  │ ├── src/ # Policy code and simulation logic 
  │ ├── a..ipynb # Single scenario notebooks 
  │ ├── b..ipynb # Multi-scenario notebooks 
  │ ├── c.multi_scenarios_final.ipynb 
  │ └── README.md 
  └── README.md # (You are here) 

  
---

## Objectives

### Phase One: Water Flow Prediction

- **Goal:** Develop robust machine learning models to predict river water flow at various stations in Brazil and France, leveraging spatio-temporal and environmental features.
- **Approach:** 
  - Data cleaning, preprocessing, and advanced feature engineering.
  - Model selection and hyperparameter optimization (Random Forest, XGB-QRF, LightGBM, EBM, MAPIE, etc.).
  - Rigorous cross-validation with spatial and temporal splits to ensure generalization.
  - Quantile regression and uncertainty estimation for robust predictions.

### Phase Two: Game Theory & Optimization

- **Goal:** Design and optimize water management policies using game theory and evolutionary algorithms.
- **Approach:**
  - Simulate water allocation scenarios under various cooperation and competition strategies.
  - Use genetic algorithms to optimize policy parameters for different objectives (e.g., fairness, efficiency).
  - Provide ready-to-use policy functions for integration and evaluation.

---

## Phase One: Water Flow Prediction

### Data Preprocessing & Feature Engineering

- Custom and baseline preprocessing pipelines.
- Feature engineering includes PCA, clustering, snow index, cyclical encoding, and more.
- All steps are documented in Jupyter notebooks under `src/notebooks`.

### Model Training & Evaluation

- Multiple model types: Quantile Random Forest (QRF), XGB-QRF, LightGBM, Explainable Boosting Machine (EBM), MAPIE.
- Hyperparameter optimization and model selection via cross-validation.
- Model saving and reproducible evaluation workflows.
- Emissions tracking for model training (CodeCarbon).

### Reproducibility

To reproduce results:

1. Ensure the `dataset` folder is present at the repository root.
2. Follow the notebook execution order in `phase_one_waterflow_prediction_machine_learning/README.md` or start from `src/notebooks/Summary`.
3. All intermediate and final results are saved in the `models/` and `data/evaluation/` directories.

---

## Phase Two: Game Theory & Optimization

### Simulation Scenarios

- **Single Scenario:** Custom, logistic, exponential, baseline, and high-cooperation policies.
- **Multi-Scenario:** Generalization and robustness analysis across multiple simulated environments.

### Policy Optimization

- Evolutionary search for optimal policy parameters.
- Ready-to-use policy functions in `src/policies/custom_final_policies.py`.
- All notebooks include precomputed results for quick review.

---

## Setup & Installation

### Python Environment

- Recommended Python version: **3.12+**
- Each phase has its own `requirements.txt` file.

### Installation Steps

```sh
# Clone the repository
git clone https://github.com/<your-org>/WaterScarcity.Hackathon.git
cd WaterScarcity.Hackathon

# Phase One setup
cd phase_one_waterflow_prediction_machine_learning
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Phase Two setup
cd ../phase_two_game_theroy_genetic_optimization
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Usage
##### Phase One
All workflows are organized as Jupyter notebooks in src/notebooks.
Start from src/notebooks/Summary for full pipeline execution.
See phase_one_waterflow_prediction_machine_learning/README.md for detailed instructions.
##### Phase Two
Notebooks prefixed with a. are for single scenario analysis.
Notebooks prefixed with b. are for multi-scenario analysis.
c.multi_scenarios_final.ipynb provides the final multi-scenario results.
See phase_two_game_theroy_genetic_optimization/README.md for scenario and policy details.


### Acknowledgements
- Capgemini Water Scarcity Hackathon 2025
- All open-source contributors and libraries used in this project