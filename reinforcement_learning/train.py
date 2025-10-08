import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from envs.gridworld import Gridworld
from rl.qlearning import train_q_learning, greedy_policy_from_Q
from rl.utils import moving_average

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=int, default=3000)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    env = Gridworld(rows=args.rows, cols=args.cols, max_steps=args.max_steps)

    Q, returns, train_success = train_q_learning(
        env,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_episodes=args.eps_decay,
        seed=args.seed,
    )

    policy = greedy_policy_from_Q(Q)

    os.makedirs(args.outdir, exist_ok=True)
    np.save(f"{args.outdir}/Q.npy", Q)
    np.save(f"{args.outdir}/policy.npy", policy)
    np.save(f"{args.outdir}/returns.npy", returns)

    smoothed = moving_average(returns, k=100)
    plt.figure()
    plt.title("Q-learning on Gridworld: Return per Episode (100-ep MA)")
    plt.xlabel("Episode")
    plt.ylabel("Return (moving avg)")
    plt.plot(smoothed)
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/learning_curve.png", dpi=150)
    print(f"Saved Q-table, policy, returns, and plot to '{args.outdir}/'")
    print(f"Training success rate: {train_success:.3f}")

if __name__ == "__main__":
    main()
