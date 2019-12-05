# pysc2-rl
Experimenting with RL frameworks and pysc2

## Requirements
* Tensorflow gpu, install on Windows with `pip install --upgrade tensorflow-gpu`
* Cuda 9.0 and CuDNN >=7.2 (see https://www.tensorflow.org/install/gpu)
* Tensorforce installed from GitHub (https://github.com/tensorforce/tensorforce)
* StarCraft II and minigame maps (see https://github.com/deepmind/pysc2)

## To Implement

* config settings for choosing how to do health and shields. (i.e. categories instead of log(health))
* refactor config file to have sections for state rep, action rep, network structure, etc.
* modify network with config file? Kind of difficult. OR can we separate out just the part with the structure into another file?
