# Project: Deep RL Quadcopter Controller

## Overview
The goal of this project is to train quadcopter with a deep reinforcement
learning algorithm such that it can achieve a take-off task. For the algorithm,
we use the Deep Deterministic Policy Gradient (DDPG).

## Contents
The key contents of this repository are as follows:

- `Quadcopter_Project.ipynb`: This Jupyter notebook provides a part of the code
for training the quadcopter and summary of our implementation and results.

- `Quadcopter_Project.html`: HTML export of `Quadcopter_Project.ipynb`. 

- `task.py`: This file defines the task (take-off). Particularly, the reward is
defined here.

- `physics_sim.py`: This file introduces a physical simulator for the motion of the
quadcopter.

- `agents/agent.py`:  This file defines the main part of the DDPG algorithm.

- `agents/actor_critic_model.py`: This file defines the actor model and critic models
used for the DDPG algorithm. These models are imported to `agent.py`.  

- `requirement.txt`: A list of the libraries used for this project.

- `plot.ipy`: This Jupyter notebook can be used to check the behavior of the quadcopter
when the cell for training DDPG agent in `Quadcopter_Project.ipynb` is still running.

- `reward.txt`: Final reward, position, orientation, velocity and angular velocity
as well as the speeds of the rotors in each episode are recorded here.


<!-- # Deep RL Quadcopter Controller

*Teach a Quadcopter How to Fly!*

In this project, you will design an agent to fly a quadcopter, and then train it using a reinforcement learning algorithm of your choice!

## Project Instructions

1. Clone the repository and navigate to the downloaded folder.

```
git clone https://github.com/udacity/RL-Quadcopter-2.git
cd RL-Quadcopter-2
```

2. Create and activate a new environment.

```
conda create -n quadcop python=3.6 matplotlib numpy pandas
source activate quadcop
```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `quadcop` environment.
```
python -m ipykernel install --user --name quadcop --display-name "quadcop"
```

4. Open the notebook.
```
jupyter notebook Quadcopter_Project.ipynb
```

5. Before running code, change the kernel to match the `quadcop` environment by using the drop-down menu (**Kernel > Change kernel > quadcop**). Then, follow the instructions in the notebook.

6. You will likely need to install more pip packages to complete this project.  Please curate the list of packages needed to run your project in the `requirements.txt` file in the repository. -->
