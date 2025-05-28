# Water Management Simulation

This repository contains all the work acheived by the team SmartWeather. You'll find mostly visualization notebooks, and the final policy. In the project structure you will find a quick description of each file objective, we highly encourage you to evaluate our phase 2 report while exploring these notebooks to get even more insights on the work acheived.
Thanks for your time, it has been a really challenging event for us and we would be pleased to discuss our results and your visions on them as it represent for us the biggest interest over scoring and ranking.

## Usage
### Single Scenario (Notebook)
"a.N"-prefix, these notebooks will contains executions of the various tests we tried including : 
 - Custom policy with linear priority factor
 - Custom policy with logistic priority factor
 - Custom policy with exponential priority factor
 - Extreme cooperation
 - Baseline model
#### Instruction
To choose wich type of execution you want please set ```USE_PRETRAINED_VALUES``` boolean value. True means that you will use already computed weights. False would mean to launch a complete EvolutionnarySearch over the desired strategy (this can take a while depending on desired n_gen and pop_size). Don't hesitate to test it, even on small values to get an idea of how our processing works.

### Multi-Scenario Analysis (Notebook)
"b.N"-prefix, these notebooks will contains generalization results of the various tests we tried including : 
 - Custom policy with linear priority factor
 - Custom policy with logistic priority factor
 - Custom policy with exponential priority factor
 - Extreme cooperation
 - Baseline model
#### Instruction
For these notebooks, optimization phase is supposed to be done so a simple "run all" should launch all cells. If you will to report any optimized weights, please set them according to scenario type using ```scenario_0_best_ind``` and ```scenario_1_best_ind``` values.
#### Note
All our notebooks will be submitted including execution results, so basically you won't even need to execute them to get your first approach to our strategies.

### Multi-Scenario Final results (Notebook)
"c.multi_scenarios_final.ipynb", if you will to only execute the multi scenarion on our best performing strategy you can execute this notebook.
#### Instruction
This one is made to be ready to use so only a "run all" is needded.

### Extracting our best performing strategy
You will probably need to take out our policies out of our environnement to execute it in your own evaluation system. For this specific reason we created "./src/policies/custom_final_policies.py" file which contains ready to use quota and incentive functions. We extracted generated weights and hard coded them to avoid any conflict during external use. Feel free to take this file in your evaluation system and let us know if you face any issues.

## Installation

```shell
# Recommended creating virtual env
   python -m venv .venv

# Activate virtual env
   ./.venv/Scripts/activate

# Install dependencies:
   pip install -r requirements.txt

# Now you are ready to execute any notebook by clicking "run all"

# Project Structure
# We explicitly gave you all our execution results for various scenarios so you can have insights on our development process
# For us, it's the most interesting part regardless of performances, so bellow is the description of each file location and goals
    ├── src/
    │   ├── policies/
    │   │   ├── custom_final_policies.py              # Ready to use quota and incentive policies
    │   │   ├── custom_linear_evol_policies.py        # Custom policy implementing priority factor computation based on linear formula (usable in EvolutionnarySearch context)
    │   │   ├── custom_logisitic_evol_policies.py     # Custom policy implementing priority factor computation based on adjustable logistic function (usable in EvolutionnarySearch context)
    │   │   ├── custom_exponential_evol_policies.py   # Custom policy implementing priority factor computation based on adjustable exponential function (usable in EvolutionnarySearch context)
    │   └── optim.py                                  # EvolutionnarySearch class, containing everything implemented to optimize our policies
    ├── a.N.single_strategy.ipynb      # Described above
    ├── b.N.multi_strategy.ipynb       # Described above
    └── c.multi_scenarios_final.ipynb  # Described above