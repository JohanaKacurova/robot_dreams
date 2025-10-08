from typing import Tuple
import numpy as np
import random

def epsilon_by_episode(ep: int, eps_start: float = 1.0, eps_end: float = 0.05, decay_episodes: int = 3000) -> float:
    if ep >= decay_episodes:
        return eps_end
    frac = ep / float(decay_episodes)
    return eps_start + (eps_end - eps_start) * frac

def choose_action(Q: np.ndarray, s: tuple, eps: float, n_actions: int) -> int:
    if random.random() < eps:
        return random.randrange(n_actions)
    r, c = s
    return int(np.argmax(Q[r, c]))

def train_q_learning(env,
                     episodes: int = 4000,
                     alpha: float = 0.1,
                     gamma: float = 0.99,
                     eps_start: float = 1.0,
                     eps_end: float = 0.05,
                     eps_decay_episodes: int = 3000,
                     seed: int = 0) -> Tuple[np.ndarray, np.ndarray, float]:
    random.seed(seed)
    np.random.seed(seed)

    Q = np.zeros((env.rows, env.cols, env.action_space), dtype=np.float32)
    returns = []
    successes = 0

    for ep in range(episodes):
        s = env.reset()
        done = False
        G = 0.0
        eps = epsilon_by_episode(ep, eps_start, eps_end, eps_decay_episodes)

        while not done:
            a = choose_action(Q, s, eps, env.action_space)
            s_next, r, done, _ = env.step(a)
            r0, c0 = s
            r1, c1 = s_next

            td_target = r + (0.0 if done else gamma * np.max(Q[r1, c1]))
            td_error  = td_target - Q[r0, c0, a]
            Q[r0, c0, a] += alpha * td_error

            s = s_next
            G += r

        if env.s == env.goal:
            successes += 1
        returns.append(G)

    success_rate = successes / episodes
    return Q, np.array(returns, dtype=np.float32), success_rate

def greedy_policy_from_Q(Q: np.ndarray) -> np.ndarray:
    rows, cols, actions = Q.shape
    pol = np.zeros((rows, cols), dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            pol[r, c] = int(np.argmax(Q[r, c]))
    return pol

def run_episode_greedy(env, Q: np.ndarray, render: bool = False):
    s = env.reset()
    done = False
    G = 0.0
    steps = 0
    while not done and steps < env.max_steps:
        r0, c0 = s
        a = int(np.argmax(Q[r0, c0]))
        s, r, done, _ = env.step(a)
        G += r
        steps += 1
        if render:
            env.render()
    return G, steps, (env.s == env.goal)

def evaluate(env, Q: np.ndarray, n_episodes: int = 20) -> dict:
    rets, steps, wins = [], [], 0
    for _ in range(n_episodes):
        G, st, ok = run_episode_greedy(env, Q, render=False)
        rets.append(G); steps.append(st); wins += int(ok)
    return {
        "avg_return": float(np.mean(rets)),
        "avg_steps": float(np.mean(steps)),
        "success_rate": wins / n_episodes
    }
