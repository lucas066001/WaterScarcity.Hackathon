# Water Management Simulation

This repository implements an agent-based evolutionary game to study water resource allocation under varying environmental and policy scenarios. It models multiple actors who decide whether to cooperate or defect in water usage, tracks ecological and economic impacts, and provides tools for running simulations and visualizing results.

## Installation

1. Clone the repository:
```shell
   git clone https://github.com/yourusername/water-management-simulation.git

2. Install dependencies:
```shell
   pip install -r requirements.txt

## Usage
### Single Scenario (Notebook)
Open single_scenario.ipynb to run and customize one simulation interactively.

### Multi-Scenario Analysis (Notebook)
Open multi_scenarios.ipynb for comparative visualizations and advanced analysis.

Project Structure
```shell
    ├── parameters/
    │   ├── data.csv               # Real riverflow time series
    │   └── scenarios/             # YAML parameter files for scenarios
    ├── src/
    │   ├── core.py                # Main WaterManagementSimulation class
    │   ├── actors.py              # ActorManager: decision-making & learning
    │   ├── water_allocation.py    # WaterAllocator: pumping & quota logic
    │   ├── ecology.py             # EcologyManager: flow & impact calculations
    │   ├── utils.py               # Helper functions (e.g., YAML loader)
    │   ├── plot_analysis.py       # Time-series plots for individual runs
    │   ├── scenarios.py           # Script to batch-run scenarios
    │   └── plot_multi_analysis.py # Impact trade-off & correlation plots
    ├── single_scenario.ipynb      # Interactive demo for one scenario
    ├── multi_scenarios.ipynb      # Comparative analysis notebook
    ├── requirements.txt           # Python dependencies
    └── README.md                  # Project overview and usage guide