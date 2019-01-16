# pysc2-rl
Experimenting with RL frameworks and pysc2

## Requirements
* Tensorflow gpu, install on Windows with `pip install --upgrade tensorflow-gpu`
* Cuda 9.0 and CuDNN >=7.2 (see https://www.tensorflow.org/install/gpu)
* Tensorforce installed from GitHub (https://github.com/tensorforce/tensorforce)
* StarCraft II and minigame maps (see https://github.com/deepmind/pysc2)

## To Run
From `tensorfoce/` run with `python -m pysc2.bin.agent --map CollectMineralShards --agent minerals_minigame.TestAgent`