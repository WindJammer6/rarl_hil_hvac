# rarl_hil_hvac
This repository contains the code for evaluating Robust Adversarial Reinforcement Learning for Human-in-the-Loop HVAC control. The project studies how reinforcement learning controllers behave when occupant comfort feedback is unreliable, including adversarial, exaggerated, and mixed feedback settings.

The main objective is to control a centralized HVAC temperature setpoint while balancing:

1. occupant thermal comfort, represented using Predicted Mean Vote (PMV), and  
2. HVAC energy usage.

The proposed method, RARL-HIL, trains a protagonist HVAC controller together with an adversarial agent that perturbs the human feedback signal during training. The goal is to make the HVAC controller more robust when deployed in environments where user feedback may be corrupted or unreliable.

---

## Repository Structure

```text
.
├── models/
│   └── RL model definitions and supporting files
│
├── saved_models/
│   └── Trained model weights
│
├── RARL_DQN_training_main.ipynb and RARL_DQN_training_main_with_exaggerated.ipynb
│   └── Trains the RARL-HIL protagonist and adversary agents
│
├── rl_baselines_training_main.ipynb
│   └── Trains standard RL baselines: DQN, PPO, and SAC
│
├── testing_rarl_vs_rl_baselines_with_adversarial.ipynb
│   └── Evaluates trained models under adversarial feedback
│
├── testing_rarl_vs_rl_baselines_with_exaggerated.ipynb
│   └── Evaluates trained models under exaggerated feedback
│
├── testing_rarl_vs_rl_baselines_with_mixed.ipynb
│   └── Evaluates trained models under mixed feedback
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
