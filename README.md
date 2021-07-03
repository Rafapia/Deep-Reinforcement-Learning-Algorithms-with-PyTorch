# Deep Reinforcement Learning Algorithms with PyTorch

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)

![ISC](https://lirp.cdn-website.com/ae9fd18b/dms3rep/multi/opt/final_white-232w.png)

This repository contains PyTorch implementations of deep reinforcement learning algorithms and environments. 
It has been forked from **p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch** and modified to learn a custom environment for use in the [Illini Solar Car](https://www.illinisolarcar.com)'s [Brizo](https://www.illinisolarcar.com/brizo) solar-powered car.

![Brizo](https://lirp-cdn.multiscreensite.com/ae9fd18b/dms3rep/multi/opt/DSC04463-cd193b44-2880w.jpg)

## **Algorithms Implemented**  

1. *Deep Q Learning (DQN)* <sub><sup> ([Mnih et al. 2013](https://arxiv.org/pdf/1312.5602.pdf)) </sup></sub>  
1. *DQN with Fixed Q Targets* <sub><sup> ([Mnih et al. 2013](https://arxiv.org/pdf/1312.5602.pdf)) </sup></sub>
1. *Double DQN (DDQN)* <sub><sup> ([Hado van Hasselt et al. 2015](https://arxiv.org/pdf/1509.06461.pdf)) </sup></sub>
1. *DDQN with Prioritised Experience Replay* <sub><sup> ([Schaul et al. 2016](https://arxiv.org/pdf/1511.05952.pdf)) </sup></sub>
1. *Dueling DDQN* <sub><sup> ([Wang et al. 2016](http://proceedings.mlr.press/v48/wangf16.pdf)) </sup></sub>
1. *REINFORCE* <sub><sup> ([Williams et al. 1992](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)) </sup></sub>
1. *Deep Deterministic Policy Gradients (DDPG)* <sub><sup> ([Lillicrap et al. 2016](https://arxiv.org/pdf/1509.02971.pdf) ) </sup></sub>
1. *Twin Delayed Deep Deterministic Policy Gradients (TD3)* <sub><sup> ([Fujimoto et al. 2018](https://arxiv.org/abs/1802.09477)) </sup></sub>
1. *Soft Actor-Critic (SAC)* <sub><sup> ([Haarnoja et al. 2018](https://arxiv.org/pdf/1812.05905.pdf)) </sup></sub>
1. *Soft Actor-Critic for Discrete Actions (SAC-Discrete)* <sub><sup> ([Christodoulou 2019](https://arxiv.org/abs/1910.07207)) </sup></sub> 
1. *Asynchronous Advantage Actor Critic (A3C)* <sub><sup> ([Mnih et al. 2016](https://arxiv.org/pdf/1602.01783.pdf)) </sup></sub>
1. *Syncrhonous Advantage Actor Critic (A2C)*
1. *Proximal Policy Optimisation (PPO)* <sub><sup> ([Schulman et al. 2017](https://openai-public.s3-us-west-2.amazonaws.com/blog/2017-07/ppo/ppo-arxiv.pdf)) </sup></sub>
1. *DQN with Hindsight Experience Replay (DQN-HER)* <sub><sup> ([Andrychowicz et al. 2018](https://arxiv.org/pdf/1707.01495.pdf)) </sup></sub>
1. *DDPG with Hindsight Experience Replay (DDPG-HER)* <sub><sup> ([Andrychowicz et al. 2018](https://arxiv.org/pdf/1707.01495.pdf) ) </sup></sub>
1. *Hierarchical-DQN (h-DQN)* <sub><sup> ([Kulkarni et al. 2016](https://arxiv.org/pdf/1604.06057.pdf)) </sup></sub>
1. *Stochastic NNs for Hierarchical Reinforcement Learning (SNN-HRL)* <sub><sup> ([Florensa et al. 2017](https://arxiv.org/pdf/1704.03012.pdf)) </sup></sub>
1. *Diversity Is All You Need (DIAYN)* <sub><sup> ([Eyensbach et al. 2018](https://arxiv.org/pdf/1802.06070.pdf)) </sup></sub>

All implementations are able to quickly solve Cart Pole (discrete actions), Mountain Car Continuous (continuous actions), 
Bit Flipping (discrete actions with dynamic goals) or Fetch Reach (continuous actions with dynamic goals). I plan to add more hierarchical RL algorithms soon.

## **Environments Implemented**

1. *SimpleISC*

## **Results**

#### 1. SimpleISC


### Usage ###

The repository's high-level structure is:
 
    ├── agents                    
        ├── actor_critic_agents   
        ├── DQN_agents         
        ├── policy_gradient_agents
        └── stochastic_policy_search_agents 
    ├── environments   
    ├── results             
        └── data_and_graphs        
    ├── tests
    ├── utilities             
        └── data structures            
