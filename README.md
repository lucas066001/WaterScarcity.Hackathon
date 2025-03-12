# WaterScarcity.Hackathon
Group participation for the 2025 hackathon organized by Capgemini. 

---
## Motivations 
- Improve overall knowledge on data science and AI
- Experiment with concrete data
- Get out of confort zone and manipulate unusual data

---
## Contributors 👥

- **[lucas066001]**
- **[Az-r-ow]**

# Hackathon on Water Scarcity 2025 - Baseline Model Repository

This repository provides a simple toolkit to train baseline models and generate submission files for the [Hackathon on Water Scarcity 2025](https://www.codabench.org/competitions/4335). The baseline models predict water discharge for the 52 stations of eval dataset. You are free to experiment with different modeling approaches or use distinct models per station, as long as your submission file adheres to the required format (see Codabench guidelines).

## Data

- **Download:**  
  Obtain the dataset from [Zenodo](https://zenodo.org/records/14826458).  
- **Setup:**  
  Unzip the dataset and place it in the root directory of the repository.

## Notebook Structure

0. **Preprocessing**
   - *01 - Data Preprocessing*
   - *02 - Feature Engineering*

1. **Training and Submission**
   - *03 - Modelisation*
   - *01 - Prediction Computation*

2. **Exploration**
   - *01 - Performance Comparison*
   - *02 - Single Model Optimisation*


## Submission

After running the notebooks, create your submission file (`data/evaluation/predictions.zip`) and upload it to [Codabench](https://www.codabench.org/competitions/4335).

## Setup Local Environment

1. **Python Version**
This repository was developed and tested using Python 3.12.6. For best compatibility, please ensure you are using Python 3.12 or a newer version.

2. **Create your venv - Mac version**

```shell
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip if needed
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Add the current directory to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$PWD"
