# Isaac Gym for Pisa SoftHand + Kinova gen3
This repository is used for reinforcement learning (RL) with [Nvidia Isaac Gym Preview 4 release](https://developer.nvidia.com/isaac-gym)


# 1) Description
This repository is a fork of [NVIDIA-Omniverse/IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) and contains additional models and training environments used for the [RePAIR project](https://www.repairproject.eu/)

For Nvidia's documentation refer to the [main branch](https://github.com/RePAIRProject/soft_hand_isaac_gym_rl/tree/main)

# 2) Installation
Requirements (tested on):
- Ubuntu 20.04.6 LTS
- Python 3.8.10

## Installation of required software
- [Nvidia Isaac Gym Preview 4 release](https://developer.nvidia.com/isaac-gym)

## Installation of IsaacGymEnvs
```
git clone -b kinova_arm_qb_hand git@github.com:RePAIRProject/soft_hand_isaac_gym_rl.git
cd soft_hand_isaac_gym_rl
pip install -e .
```

# 3) Usage
Go to the folder isaacgymenvs
```
cd isaacgymenvs
```

To start a new training 

```
python3 train.py task=SoftGrasp headless=True
```

To continue a training

```
python3 train.py task=SoftGrasp headless=True checkpoint=runs/[folder]/nn/[file].pth
```

To test 

```
python3 train.py task=SoftGrasp test=True checkpoint=runs/[folder]/nn/[file].pth num_envs=64
```

# 4) Known Issues
None.

# 5) Relevant publications
```
@misc{makoviychuk2021isaac,
      title={Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning}, 
      author={Viktor Makoviychuk and Lukasz Wawrzyniak and Yunrong Guo and Michelle Lu and Kier Storey and Miles Macklin and David Hoeller and Nikita Rudin and Arthur Allshire and Ankur Handa and Gavriel State},
      year={2021},
      journal={arXiv preprint arXiv:2108.10470}
}
```
