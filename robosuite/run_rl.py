import argparse
import os, glob, shutil
import numpy as np

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.logger import pretty_print

import robosuite as suite
from robosuite.wrappers import GymWrapper

"""
Code is updated for Ray=2.2.0. 
"""


def setup_config(algo_name, env, seed=0):
    if algo_name == 'ppo':
        # TODO: Not implemented
        config = None
    elif algo_name == 'sac':
        config = (SACConfig()
                  .rollouts(num_rollout_workers=4)
                  .resources(num_gpus=0)
                  .environment(env=env)
                  .training(gamma=0.9, lr=0.01)
                  .debugging(seed=seed)
                  )
    return config


def load_policy(algo_name, env, env_name, policy_path=None, seed=0, extra_configs={}):
    if algo_name == 'ppo':
        pass
    elif algo_name == 'sac':
        algo = setup_config(algo_name, env, seed).build()
    if policy_path != '':
        if 'checkpoint' in policy_path:
            algo.restore(policy_path)
            print("##################")
            print("Loading directly from a specific policy path:", policy_path)
            print("##################")
        else:
            # Find the most recent policy in the directory
            directory = os.path.join(policy_path, algo, env_name)
            files = [f.split('_')[-1] for f in glob.glob(os.path.join(directory, 'checkpoint_*'))]
            files_ints = [int(f) for f in files]
            if files:
                checkpoint_max = max(files_ints)
                checkpoint_num = files_ints.index(checkpoint_max)
                checkpoint_path = os.path.join(directory, 'checkpoint_%s' % files[checkpoint_num], 'checkpoint-%d' % checkpoint_max)
                algo.restore(checkpoint_path)
                print("##################")
                print("Inferring policy to load based on env_name:", checkpoint_path)
                print("##################")

                # return agent, checkpoint_path
            return algo, None
    return algo, None


def make_env(env_name, seed=0):
    env = GymWrapper(
        suite.make(
            env_name,
            robots="Jaco",  # use Jaco robot -- TODO: can we leave this argument out?
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            has_renderer=False,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards -- TODO: change this?
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
        )
    )
    env.seed(seed)
    return env


def train(env_name, algo_name, epochs=0, save_dir='./trained_models/', load_policy_path='', seed=0, save_checkpoints=False):
    env = make_env(env_name, seed)
    algo, checkpoint_path = load_policy(algo_name, env, env_name, load_policy_path, seed)

    for i in range(epochs):
        result = algo.train()
        print(pretty_print(result))

        if not (save_checkpoints and result['training_iteration'] % 10 == 1):
            # Delete the old saved policy
            if checkpoint_path is not None:
                shutil.rmtree(os.path.dirname(checkpoint_path), ignore_errors=True)

        # Save the recently trained policy
        checkpoint_path = algo.save(os.path.join(save_dir, algo, env_name))

    return checkpoint_path


def evaluate_policy(env_name, algo_name, policy_path, n_episodes=100, seed=0, verbose=False):
    env = make_env(env_name, seed)
    algo, _ = load_policy(algo_name, env, env_name, policy_path, seed)

    rewards = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        reward_total = 0.0
        while not done:
            action = algo.compute_single_action(obs)
            obs, reward, done, info = env.step(action)

            reward_total += reward

        rewards.append(reward_total)
        if verbose:
            print('Reward total: %.2f' % (reward_total))

    print('\n', '-'*50, '\n')
    print('Reward Mean:', np.mean(rewards))
    print('Reward Std:', np.std(rewards))

    return np.mean(rewards), np.std(rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL for MuJoCo envs')
    parser.add_argument('--env', default='Reacher-v2',
                        help='Environment to train on (default: Reacher-v2)')
    parser.add_argument('--algo', default='ppo',
                        help='Reinforcement learning algorithm')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train a new policy')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Whether to render a single rollout of a trained policy')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Whether to evaluate a trained policy over n_episodes')
    parser.add_argument('--train-epochs', type=int, default=100,
                        help='Number of simulation epochs to train a policy (default: 100)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='Directory to save trained policy in (default ./trained_models/)')
    parser.add_argument('--load-policy-path', default='./trained_models/',
                        help='Path name to saved policy checkpoint (NOTE: Use this to continue training an existing policy, or to evaluate a trained policy)')
    parser.add_argument('--render-episodes', type=int, default=1,
                        help='Number of rendering episodes (default: 1)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Whether to output more verbose prints')
    parser.add_argument('--save-checkpoints', action='store_true', default=False,
                        help='Whether to save multiple checkpoints of trained policy')
    args = parser.parse_args()

    if args.train:
        checkpoint_path = train(args.env, args.algo, epochs=args.train_epochs, save_dir=args.save_dir, load_policy_path=args.load_policy_path, seed=args.seed, save_checkpoints=args.save_checkpoints)
    if args.evaluate:
        evaluate_policy(args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, n_episodes=args.eval_episodes, seed=args.seed, verbose=args.verbose)
