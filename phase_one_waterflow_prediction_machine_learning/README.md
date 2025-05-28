# WaterScarcity.Hackathon

Group participation for the 2025 hackathon organized by Capgemini.

---

## Contributors 👥

- **[lucas066001]**
- **[Az-r-ow]**

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
```

## Results Reproduction

To reproduce the results that we got for the project, your entrypoint should be the `src/notebooks/Summary` folder where you will find everything you need to reproduce the results.

> For the notebooks to run correctly, you need the `dataset` folder in the root of the repository.

- `01.1 - Custom Preprocessing.ipynb`: This notebook contains the _baseline_ + custom preprocessing steps that we applied to the dataset.
- `01.2 - Baseline Preprocessing.ipynb`: This notebook contains the baseline preprocessing steps (as originally provided).
- `02.1 - Custom Feature Engineering.ipynb`: This notebook contains the custom feature engineering steps that we applied to the dataset.
- `02.2 - Baseline Feature Engineering.ipynb`: This notebook contains the baseline feature engineering steps (as originally provided).
- `03 - Model Training.ipynb`: Training of the QRF and XGBQRF model and saving them for predictions.
- `04 - Prediction Computation.ipynb`: Predictions using the `XGBQRF` models and saving the results in `data/evaluation/custom/xgb_qrf`.
- `05 - Dataset Comparison.ipynb` : Comparison of the custom dataset with the baseline dataset and tracking carbon emissions.

### Steps to get XGBQRF evaluation results

> Run the notebooks from top to bottom using "Run All Cells"

**Notebooks to run:**

- `01.1 - Custom Preprocessing.ipynb`
- `02.1 - Custom Feature Engineering.ipynb`
- `03 - Model Training.ipynb`
- `04 - Prediction Computation.ipynb`
- `05 - Dataset Comparison.ipynb`

Then you should be able to find the results in `data/evaluation/custom/xgb_qrf`.

### Steps to get the dataset comparison results

**Notebooks to run:**

- `01.1 - Custom Preprocessing.ipynb`
- `01.1 - Baseline Preprocessing.ipynb`
- `02.1 - Custom Feature Engineering.ipynb`
- `02.2 - Baseline Feature Engineering.ipynb`
- `05 - Dataset Comparison.ipynb`
