Deep Q-Network (DQN) Agent for CartPole-v1
📌 Project Overview

This project implements a Deep Q-Network (DQN) agent from scratch using PyTorch to solve the CartPole-v1 environment from OpenAI Gym (Gymnasium).

The agent learns to balance a pole on a moving cart using reinforcement learning techniques including:

Experience Replay Buffer

Target Network Updates

Epsilon-Greedy Exploration

Multi-seed training for stability analysis

The goal is to reach a high and stable average reward over time.

🛠 Technologies Used

Python 3.x

Gymnasium (CartPole-v1)

PyTorch

NumPy

Matplotlib (for plotting in notebook)

📂 Project Structure
DQN_CARTPOLE/
│
├── saved_models/
│   ├── best_seed_42.pth
│   ├── best_seed_7.pth
│   └── best_seed_123.pth
│
├── src/
│   ├── agent.py
│   ├── model.py
│   ├── replay_buffer.py
│   └── evaluate_dqn.py
│
├── train_dqn.py
├── notebook.ipynb
├── rewards_seed_42.npy
├── rewards_seed_7.npy
├── rewards_seed_123.npy
└── readme.md

🧠 DQN Implementation Details
1️⃣ Neural Network

A fully connected feedforward network is used to approximate Q-values:

Input: 4 state values

Hidden Layers: Fully connected layers with ReLU activation

Output: 2 Q-values (Left / Right actions)

2️⃣ Experience Replay

A replay buffer stores past experiences:

(state, action, reward, next_state, done)


Random mini-batches are sampled during training to break correlation between consecutive experiences.

3️⃣ Target Network

A separate target network is updated periodically to stabilize learning.

Step-based target update

target_update_steps = 200

4️⃣ Epsilon-Greedy Policy

Exploration is controlled using epsilon decay:

Initial epsilon = 1.0

Decay rate = 0.995

Minimum epsilon = 0.01

This balances exploration and exploitation.

5️⃣ Reproducibility

To ensure fair evaluation and stability analysis:

Fixed random seeds were used

Models trained on 3 different seeds:

42

7

123

This demonstrates how DQN performance varies with initialization.

🚀 How to Run the Project
1️⃣ Install Dependencies
pip install gymnasium[classic-control]
pip install torch
pip install numpy
pip install matplotlib

2️⃣ Train the Model

Open train_dqn.py.

Set the seed manually inside the file:

seed = 42


Then run:

python train_dqn.py


To train on other seeds:

Change the seed value

Run the script again

Each run saves:

saved_models/best_seed_<seed>.pth
rewards_seed_<seed>.npy

3️⃣ Evaluate a Trained Model

Run:

python src/evaluate_dqn.py


You will be prompted to enter the seed number:

Enter the seed number you want to evaluate:


The agent will run in a visible CartPole environment.

4️⃣ Plot Training Results

Open:

notebook.ipynb


The notebook:

Loads reward files

Plots episode rewards

Compares convergence behavior across seeds

Analyzes training stability

📊 Observations from Multi-Seed Training

Different seeds showed different convergence speeds.

Some seeds achieved faster stabilization.

Late-stage instability was observed in certain runs.

Reinforcement learning performance is sensitive to initialization and randomness.

This highlights the importance of multi-seed evaluation in reinforcement learning experiments.

🏁 Final Performance

The agent was able to:

Achieve high episode rewards (close to 500)

Maintain stable performance across multiple evaluation episodes

Demonstrate learning behavior consistent with DQN algorithm

🎯 Key Takeaways

Target network updates significantly improve stability.

Experience replay reduces variance.

Exploration strategy strongly impacts convergence speed.

Multi-seed training provides more reliable evaluation than single-run results.

📚 References

Mnih et al., “Human-level control through deep reinforcement learning” (DQN Paper)

OpenAI Gym Documentation

PyTorch Documentation

✅ Deliverables Included

✔ train_dqn.py – Training script
✔ evaluate_dqn.py – Model evaluation script
✔ notebook.ipynb – Reward visualization and analysis
✔ Saved trained models
✔ Reward history files
✔ This README with complete instructions

👨‍💻 Author

Implemented and experimented as part of a reinforcement learning project to understand DQN behavior and training stability in controlled environments.
