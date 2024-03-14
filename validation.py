import os
from collections import deque
import torch
import torch.nn as nn

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_test_args
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from place_env import place_envs
from rnd import RNDModel
import torch.optim as optim

args = get_test_args()

rnd = RNDModel((1, 1, args.grid_num, args.grid_num), args.grid_num * args.grid_num, args.device)
forward_mse = nn.MSELoss(reduction='none')
if args.task == 'place':
    optimizer = optim.Adam(rnd.predictor.parameters(), lr=5e-6)
elif args.task == 'fullplace':
    optimizer = optim.Adam(rnd.predictor.parameters(), lr=2e-6)


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    # device = torch.device("cuda:0" if args.cuda else "cpu")
    device = torch.device(args.device)

    if args.task == 'place':
        envs = place_envs(args.benchmark, args.grid_num, args.overlap)
        actor_critic = torch.load("./trained_models/placement_300.pt")[0]
        actor_critic.to(device)

    num_steps = args.num_mini_batch * envs.steps

    # agent = algo.PPO(
    #         actor_critic,
    #         args.clip_param,
    #         args.ppo_epoch,
    #         args.num_mini_batch,
    #         args.value_loss_coef,
    #         args.entropy_coef,
    #         lr=args.lr,
    #         eps=args.eps,
    #         max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(num_steps, args.num_processes,
                              envs.obs_space, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    # episode_rewards = deque(maxlen=10)

    features = torch.zeros(envs.steps, 2)

    for step in range(num_steps):
        # Sample actions
        n = len(envs.results)
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                rollouts.masks[step], features, n)

        # Obser reward and next obs
        obs, done, reward = envs.step(action, rnd, forward_mse, optimizer)
        features[n][0] = action // args.grid_num
        features[n][1] = action % args.grid_num

        if done:
            obs = envs.reset()
            features = torch.zeros(envs.steps, 2)
            print(reward)


if __name__ == "__main__":
    main(args)
