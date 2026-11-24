# Week 1: Reinforcement learning foundations

**Goal:** understand why the policy gradient works, and train your first agent with it.
**Time:** 8–10 hours.

## Learn (in this order)

1. **OpenAI Spinning Up, Part 1–3** (~2 h): what RL is, key concepts, and the simplest policy gradient.
   https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
2. **Lilian Weng, "Policy Gradient Algorithms"**: read up to and including REINFORCE and actor-critic (~1 h).
   https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
3. **Sutton & Barto, Chapter 13 (Policy Gradient Methods)**: free online; 13.1–13.4 are enough (~1.5 h).
   http://incompleteideas.net/book/the-book-2nd.html
4. **David Silver's RL course, Lecture 7 (Policy Gradient)**: optional, if you prefer video.
   https://www.davidsilver.uk/teaching/
5. **New to PyTorch?** Do the 60-minute blitz first.
   https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

Key ideas to be able to explain by the end of the week:

- MDPs, returns and discounting
- The policy gradient theorem and the log-derivative trick
- Why REINFORCE has high variance, and why subtracting a baseline keeps the gradient unbiased but lowers variance
- The advantage function A(s, a) = Q(s, a) − V(s)

## Build

`reinforce_cartpole.py` already trains a vanilla REINFORCE agent on CartPole-v1:

```
pip install torch gymnasium matplotlib
python reinforce_cartpole.py
```

Your tasks (marked `TODO` in the file):

1. Make `run_episode` also return the visited states.
2. Add a small value network V(s) with its own optimiser.
3. Use `returns − V(s_t)` as the advantage, and fit V(s) to the returns with an MSE loss.
4. Run both versions with 3 seeds each and plot the learning curves on one figure.

## Deliverable (due end of Week 1)

- Your code and the learning-curve plot, pushed to your fork of this repo
- A half-page note: did the baseline speed up learning or reduce the spread across seeds? Why should it, mathematically?

## Checkpoint questions (we discuss these at the weekly sync)

- Why do we multiply log π(a|s) by the return, and not π(a|s) itself?
- What happens to learning if you remove the return normalisation in the starter code?
- In Week 3, GRPO uses the *mean reward of a group of samples* as its baseline. How is that related to what you built this week?
