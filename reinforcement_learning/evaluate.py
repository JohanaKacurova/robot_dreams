import argparse
import numpy as np
from envs.gridworld import Gridworld
from rl.qlearning import evaluate, greedy_policy_from_Q

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--qpath", type=str, default="outputs/Q.npy")
    parser.add_argument("--render", action="store_true", help="Print greedy policy arrows on grid.")
    args = parser.parse_args()

    env = Gridworld(rows=args.rows, cols=args.cols, max_steps=args.max_steps)

    Q = np.load(args.qpath)
    stats = evaluate(env, Q, n_episodes=20)

    print("=== EVALUATION (20 greedy episodes) ===")
    print(f"Avg return: {stats['avg_return']:.2f}")
    print(f"Avg steps : {stats['avg_steps']:.2f}")
    print(f"Success % : {stats['success_rate']*100:.1f}%")

    if args.render:
        policy = greedy_policy_from_Q(Q)
        print("\nGreedy policy arrows (after training):")
        env.reset()
        env.render(policy=policy)

print(">>> evaluate.py started")

if __name__ == "__main__":
    main()
    print(">>> evaluate.py done")
