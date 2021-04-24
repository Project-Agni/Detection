<div align="center">

# 🔥 Detection 🔥
🧠 An Unsupervised Reinforcement Learning Pipeline for Video Frame Classification

| **🚧 This is a Proof of Concept Project 🚧** |
|:-------------------:|

| **🚧 Authors are not Responsible for Damages to Life and Property if Deployed 🚧** |
|:-------------------:|

---

</div>

# Usage 👨‍💻
Get the dataset from [here](https://drive.google.com/drive/folders/1HznoBFEd6yjaLFlSmkUGARwCUzzG4whq?usp=sharing) and 
place it under datasets. 
```shell
python runner.py --arch [cnn, dqn, usrl]
```
The trained weights will be stored in the root of the runner script. 

## Inference
```shell
python test.py
```

# Todo 📜
- [x] CNN
- [x] RL - DQN
- [x] RL - USRL
- [x] Live cam test script

# References 📑

- [Creating a custom gym env](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e)
- [Boilerplate for trainer scripts](https://github.com/pytorch/examples/blob/master/mnist/main.py)
- [DQN Implementation](https://github.com/Syzygianinfern0/Stable-Baselines)
- [Unsupervised State Representation Learning](https://github.com/mila-iqia/atari-representation-learning)
- [Project Inspiration](https://github.com/arpit-jadon/FireNet-LightWeight-Network-for-Fire-Detection)