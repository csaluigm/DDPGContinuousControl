# DDPG Agent
DDPG agent for the Unity Reacher environment

![](agent.gif)
# Project Details

The environment of this project has 4 possible actions corresponding to torque applicable to two joints (each with value in [-1,1]).
The state space has 33 dimensions and contains position, rotation, velocity, and angular velocities of the arm
The environment reward system is quite simple, +0.1 each time the arm touches the target.
The environment is considered solved when the agent achieves an average reward of +30 (over 100 consecutive episodes).

# Getting Started

Make sure you have python installed in your computer. (The project was created with python 3.6.12) [download python](https://www.python.org/downloads/)

Navigate to the root of the project:

`cd ContinuousControlDDPG` 

Install required python packages using pip or conda, for a quick basic setup use:

`pip install -r requirements.txt` 

The repo already contains the windowsx64 version of the unity environment otherwise you would need to download it and place it under the Unity folder.

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

# Instructions

You can run the project from some Editor like VS code or directly from commandline:

`python main.py --train 1`

This will train the agent and will store 2 versions of model weights. One when it pass the environment solved condition and other after the training episodes.

it stores the model in actor_trained_model.pth and critic_trained_model.pth on the root of the project.

Once the model is trained you can check its behavior by testing it:

`python main.py`