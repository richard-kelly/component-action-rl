# pysc2-rl
Experimenting with RL frameworks and pysc2

## Requirements
* Tensorflow gpu, install on Windows with `pip install --upgrade tensorflow-gpu`
* Cuda 9.0 and CuDNN >=7.2 (see https://www.tensorflow.org/install/gpu)
* Tensorforce installed from GitHub (https://github.com/tensorforce/tensorforce)
* StarCraft II and minigame maps (see https://github.com/deepmind/pysc2)

## To Implement

* An easier way to run a batch of runs in one map with one config, followed by another batch on the same learned models to do curriculum learning.
* logging of action choice stats. We need to know which actions the thing chooses, and how often, and how that changes over time.
* config settings for choosing how to do health and shields. (i.e. categories instead of log(health))
* refactor config file to have sections for state rep, action rep, network structure, etc.
* modify network with config file? Kind of difficult. OR can we separate out just the part with the structure into another file?
