[//]: # "Image References"

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# DDPG (Actor/Critic) Reinforcement Learning using PyTorch and Unity ML -  Reacher Environment 

### Overview

This project was developed as part of Udacity Deep Reinforcement Learning Nanodegree course. This project solves Reacher environment by training the agent to control robotic arms by positioning them in moving target locations using Deep Deterministic Policy Gradient (DDPG) algorithm. The environment is based on [Unity ML agents](https://github.com/Unity-Technologies/ml-agents). 

### Introduction

For this project, the unity ML [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment is used . Agent is trained to learn to control l robotic arms to position them in moving target locations

![Trained Agent][image1]

The environment contains 20 identical agents, each with its own copy of the environment. The environment  takes into account the presence of many agents.  In particular, agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,

- After each episode, the rewards that each agent received (without discounting),  is used to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

### Rewards:

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

### Environment:

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.



### Getting Started

### Installation and Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

   - **Linux** or **Mac**:

   ```
   conda create --name drlnd python=3.6
   source activate drlnd
   ```

   - **Windows**:

   ```
   conda create --name drlnd python=3.6 
   activate drlnd
   ```

2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.

   - Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
   - Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).

3. Clone the repository (if you haven't already!), and navigate to the `python/` folder. Then, install several dependencies.

```
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

1. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.

```
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

1. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.

[![Kernel](https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png)](https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png)



### Unity Environment Setup:

Unity Environment is already built and made available as part of Deep Reinforcement Learning course at Udacity.

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

### Deep Deterministic Policy Gradient Algorithm

DDPG is a model-free off-policy actor-critic algorithm that learns directly from observation spaces. DDPG employs Actor-Critic model, where Actor learns the policy and Critic learns the value function to evaluate the quality of the action chosen by the policy. while Deep Q-Network learns the Q-function using experience replay  and works well in discrete space, DDPG algorithm extends it to  continuous action spaces using Actor-Critic framework while learning policy.

![image-20200403165854862](images\image-20200403165854862.png)

### Repository

The repository contains the below files:

- ddpg_continuous_agent.ipynb :  Model for Actor and Critic along with agent learns using Experience Replay and OUNoise. Training the agent and testing the agent is implemented here.
- ddpg_continuous_agent_prioritized_replay.ipynb : Model for Actor and Critic along with agent learns using Prioritized Experience Replay and OUNoise. Training the agent and testing the agent is implemented here. The agent trained using DDPG with prioritized experience replay buffer learned faster. 
- checkpoint_actor.pth : Learned model weights for Actor
- checkpoint_critic.pth : Learned model weights for Critic
- checkpoint_actor_preplay.pth : Learned model weights for Actor with Prioritized replay
- checkpoint_critic_preplay.pth : Learned model weights for Critic with Prioritized replay
- images  directory: contains images used in documentation
- multiple_agents: Please copy Reacher environment [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip) to this location or modify the filepath in ddpg_continuous_agent.ipynb to point to the correct location.



## Model Architecture:

Pendulum-v0 environment with [Deep Deterministic Policy Gradients (DDPG)](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/DDPG.ipynb) is used as reference  to build the model.  The model architecture that is used is:

Actor:
	Input(state size of 32) &rarr; Dense Layer(256) &rarr; RELU &rarr; Dense Layer(128) &rarr; RELU &rarr; Dense Layer( action size of 4) &rarr; TANH

Critic:
	Input(state size of 32) &rarr; Dense Layer(256) &rarr; RELU &rarr; Dense Layer(128) &rarr; RELU &rarr; Dense Layer( action size of 4) 

Agent:
	Actor Local and Critic Local networks are trained and updates the Actor Target and Critic Target networks using weighting factor Tau.

Please refer to Report.md for more details on the model and parameters used for tuning

## Results:

Results from DDPG training  with prioritized experience replay and experience replay buffer are shared below. The agent trained using prioritized experience replay showed faster and efficient learning than experience replay. The agent with DDPG with prioritized experience replay learned the environment in 24 less episodes than the agent with DDPG and experience replay.

##### DDPG with Prioritized Experience Replay

![image-20200405013049322](D:\DeepLearning\git\ddpg_reinforcement_learning\images\image-20200405013049322.png)

##### DDPG with Experience Replay

![image-20200403170122634](D:/DeepLearning/git/ddpg_reinforcement_learning/images/image-20200403170122634.png)

