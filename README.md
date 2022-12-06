## Replication of "Optimising Lockdown Policies for Epidemic Control using Reinforcement Learning" 

Overview: This repository replicates the above described [paper](https://arxiv.org/abs/2003.14093) from Harshad Khadilkar, Tanuja Ganu, Deva P Seetharam. Below contains the brief instruction to run this github replication code. 

### Notes to DSC180A TA's
The replication doesn't contain actual data, given that it's an "online" COVID simulator that varies based on the policy of the reinforcement learning model. In other words, the data is fixed and variable to the actions performed by the model. Last but not least, there is no testing data, given that the task is to maximize the reward and find the optimal policy given a duration, not an inferennnce tasks

### Instructions - Conda Virtual Environment
In this section, you'll execute the code with the below steps:
1. Create a conda environment with python version 3.9 `conda create --name placeholder_name python=3.9`. Note the "placeholder_name" is the environment name that you desire
2. Activate the conda environment `conda activate placeholder_name`. 
3. Within the environment, install the python packages by running `pip install -r full_requirements.txt`
4. By this stage, the conda environment should contain all of the required packages. To execute the code, run `python main.py` (or `python3 main.py`)

## Repository Structure
- The repository currently contains config folder, models folders, and utils folder. 
- The config folder will store all of the hardcoded constants and allows one to tune and perform the experiments/hyperparameters. While the files have been created, it has not been integrated into the code yet
- The models folder contain all of the model components of the DeepQ network and the simulator. 
- The utils folder contains helper methods, utility-based files, and miscallaneous files. 
- main.py performs the execution of the code. Therefore, you'll be compiling on main.py

README update date: December 5th, 2022