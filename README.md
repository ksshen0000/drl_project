# SAC-based hybrid method

This repository is the implementation of baseline models and our hybrid models. Note that we didn't upload all the saved models into the repository because it's too large. We upload the saved hybrid model for humanoid. The current outcome is that the humanoid will try to stand up from lying down but only can sit up right now due to limited computing resources.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Run Commands](#run-commands)
- [Command-Line Arguments](#command-line-arguments)
- [Project Structure](#project-structure)

---

## Environment Setup

### Requirements

- **Operating System**: Windows/Linux/macOS
- **Programming Language**: Python 3.8+ (or specify another language)
- **Dependencies**:  
  - List all required libraries and versions, e.g.,
    - `numpy >= 1.20.0`
    - `requests >= 2.25.0`

### Optional Tools

- **IDE**: VS Code, PyCharm, or any preferred text editor.
- **Version Control**: Git 2.30+.

---

## Run Commands
### demo
`python SAC_TRPO.py --render --load`
`python SAC_MC.py --render --load`

### plot
`python plot.py -env_name [env]`
Note this command need the data after training.


## Command-Line Arguments

- `--env_name`: Can be any gym environment name. In this case, we use `HumanoidStandup-v2` as default.
- `--train_eps`: total episodes in training loop.
- `--max_steps`: max_steps to go to into next loop. Prevent infinite loop.
- `--load`: load local model. No extra argument(store_true).
- `--save_interval`: interval for saving model to disk.
- `--render`: enable render the GUI animation. No extra arguments.

## Project Structure
- `plot.py`: plot figure of EWMA results.
- `figure/`: store the figures.
- `models/`: save the model for each algorithm environment pair.
- `[algorithm].py`: implementation of each algorithm.