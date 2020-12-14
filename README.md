# component-action-rl
This project contains implementations of reinforcement learning algorithms for real-time strategy (RTS) game environments in which the action space can be viewed as a combination of multiple components (e.g. perform action X at position Y). It was made for my master's research.

There is a component for use with Deepmind's [pysc2](https://github.com/deepmind/pysc2) component of the StarCraft II Learning Environment, which gives a Python API for interacting with the game, designed for RL research.

There is also a component for use with microRTS, a toy RTS game designed for AI research. I started working with microRTS for a while in the middle of work on pysc2, and decided not to continue with it at the time. In this project is a Python RL part using TensorFlow, sharing code with the pysc2 work. It communicates via socket with a Java module in microRTS.  

## Requirements
* Python 3.6.8
* Tensorflow gpu v1.14, install with `pip install --upgrade tensorflow-gpu==1.14.0`
* Cuda 10.0 and corresponding CuDNN 7.x (see https://www.tensorflow.org/install/gpu)
* StarCraft II and pysc2 Python package for StarCraft II work
* microRTS (my fork [here](https://github.com/richard-kelly/microrts)) for microRTS work

## How to run
Full documentation for how to use everything here is **under construction**. 

* Run scripted bots with eval_dir mode, inference_only on, and a scripted bot selected. Results will go in the folder specified as model_dir
* Run models in eval mode with different maps by specifying the maps and a new model dir in an experiment file, running in experiment mode, having inference_only on. 