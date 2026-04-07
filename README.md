# Non-linear Switched Martin-Sarras Observer

## The Project
In this project the attitude and IMU (accelerometer and gyroscope) biases are estimated using a set of 3 non-linear observers.
Each observer has conditions under which it can be used and provably converges to the truth. 
The decision of when to use each observer is difficult to ascertain a priori despite having clear conditions (persistent excitement and validity of accelerometer).
For this reason, we use an offline planning method of branch and bound tree search to take a trajectory and find the near-optimal policy per trajectory. 
These labels will be used to train a neural network to decide online in real-time which observer to use.

## What I tried
At first I tried to use Q-learning to directly learn when each should be used.
The reward function was straight forward, but even trying to overfit one trajectory was taking over an hour, so trying to generalize looked like would take much longer than we had.
Instead, I chose to try and fine an offline classical method to find the optimum.
After significant deliberation, branch and bound seemed like a good choice to find the optimum because it can be fast and is near optimal.
Also, since the trajectory was pregenerated we could easily just rollout the estimation which is extremely cheap.

## Current Solution
Using branch and bound we search for the best option over the next 7 decsions (decisions are made at 4hz) and then roll forward to the next decision and then search the next 7 deep.
Reward shaping became an important portion of this, because errors in estimation are a non-issue when they are low enough, but a big deal when large.
So non-linear gains on the estimation errors is employed, so when errors go above a threshold (4 degrees) they are penalized much more.
This allows the observer to estimate "less observable" states while the attitude estimate is still "good enough".
I was able to show that this method beats the mainstay complimentary filter developed by Mahony.  
Though it can't run in real time it can be used to generate labels for a future ML approach that could run in real time.

## How To Run
The given script generates a trajectory from a custom simulation.
It then runs the system on that trajectory and outputs a suite of plots covering various error statistics.
It also compares my observer's output to the Mahony filter.

First, install the dependencies,

```bash
git clone --recursive https://github.com/iandareid/switched-Martin-Sarras-observer.git
cd switched-Martin-Sarras-observer
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy matplotlib PyYAML mujoco jax flax
```

If you already cloned the repo without submodules, run:

```bash
git submodule update --init --recursive
```

Next, run the plot generation script.

```bash
python3 observer_labeling/scripts/preview_and_label_trajectory.py --target-depth 7
```

You should see a window pop up with the simulator creating the 25 second trajectory.
Once you close the window, the branch and bound algorithm will then run, this will likely take over 400 seconds to run. You will then find the plots generated in the `results` directory.
