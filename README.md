# Dots And Boxes Agent

This repository contains our Dots and Boxes agent developed for **HAIC 2025 (Hanyang Artificial Intelligence Challenge)**.  
We achieved **2nd place** in the competition.

Our agent’s strength comes mainly from a **search algorithm**, 
and this repo is a **refactored version** of the experimental environment we used during the challenge.

## How to run our code
**environment setting**
``` bash
pip install -r requirements.txt
# Tested with Python 3.10
```

**How to run experiments**
``` bash
python run_experiments.py [CONFIG_PATH or CONFIG_DIR]
```

Example:

- If you pass **a JSON file,** it runs that experiments only:
``` bash
python run_experiments.py config/exp_configs/Basic_vs_d2~20_extension.json
```

- If you pass **a directory**, it runs all *.json configs in that directory:
``` bash
python run_experiments.py config/exp_configs
```
Example:
```bash
python play_with_ai.py configs/policy_configs/d2~25.json
```
**How to Play against our AI**
``` bash
python play_with_ai.py [POCLICY_CONFIG_PATH]
```
POLICY_CONFIG is also a **single JSON file**
If it runs Successfully, GUI based on pygame will show up.

## Game Environment
environment is same as the challenge has provided

- **Board size**: **5 x 5 Boxes**(6 x 6 dots)
<img src="assets/Dots_And_Boxes_ex.png" width="400" alt="Dots and Boxes example board" />

- **Time Limit**: **24 seconds** per game
- **Hardware**: **CPU only** Environment. GPU is not allowed

## Future Work / Possible Improvements

**1. Stronger and more principled heuristics**
- Our experiments show that most of the additional handcrafted heuristics we tried doesn't show much improvement.
- So For improvement. We need to explore:
    - **Better mid-game evaluation** (e.g., chain/junction heuristics)
    - **Tighter integration between search and Dots and Boxes theory**

**2. Neural Network–based agent**

- Due to the CPU-only constraint in the challenge environment, we could not - fully utilize GPU-based neural networks.
- However, prior work such as **AlphaZero** and **MuZero** suggests that NN-based methods can perform extremely well.
- We would like to:
    - Train a value/policy network for Dots and Boxes
    - Compare a **pure search-based agent vs a NN-augmented search agent** (AlphaZero-style)