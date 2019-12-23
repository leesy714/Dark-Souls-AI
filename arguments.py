import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='DQN')

    # Basic Arguments
    parser.add_argument('--seed', type=int, default=1122,
                        help='Random seed')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # Training Arguments
    parser.add_argument('--max-frames', type=int, default=1400000, metavar='STEPS',
                        help='Number of frames to train')
    parser.add_argument('--buffer-size', type=int, default=10000, metavar='CAPACITY',
                        help='Maximum memory buffer size')
    parser.add_argument('--update-target', type=int, default=500, metavar='STEPS',
                        help='Interval of target network update')
    parser.add_argument('--train-freq', type=int, default=1, metavar='STEPS',
                        help='Number of steps between optimization step')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='γ',
                        help='Discount factor')
    parser.add_argument('--learning-start', type=int, default=1000, metavar='N',
                        help='How many steps of the model to collect transitions for before learning starts')
    parser.add_argument('--eps_start', type=float, default=1.0,
                        help='Start value of epsilon')
    parser.add_argument('--eps_final', type=float, default=0.0,
                        help='Final value of epsilon')
    parser.add_argument('--eps_decay', type=int, default=30000,
                        help='Adjustment parameter for epsilon')
    parser.add_argument('--train-during-reset', action='store_true',
                        help='Train during reset')

    # Algorithm Arguments
    parser.add_argument('--double', action='store_true',
                        help='Enable Double-Q Learning')
    parser.add_argument('--dueling', action='store_true',
                        help='Enable Dueling Network')
    parser.add_argument('--noisy', action='store_true',
                        help='Enable Noisy Network')
    parser.add_argument('--prioritized-replay', action='store_true',
                        help='enable prioritized experience replay')
    parser.add_argument('--c51', action='store_true',
                        help='enable categorical dqn')
    parser.add_argument('--multi-step', type=int, default=1,
                        help='N-Step Learning')
    parser.add_argument('--Vmin', type=int, default=-100,
                        help='Minimum value of support for c51')
    parser.add_argument('--Vmax', type=int, default=100,
                        help='Maximum value of support for c51')
    parser.add_argument('--num-atoms', type=int, default=51,
                        help='Number of atom for c51')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha value for prioritized replay')
    parser.add_argument('--beta-start', type=float, default=0.4,
                        help='Start value of beta for prioritized replay')
    parser.add_argument('--beta-frames', type=int, default=100000,
                        help='End frame of beta schedule for prioritized replay')
    parser.add_argument('--sigma-init', type=float, default=0.5,
                        help='Sigma initialization value for NoisyNet')

    # Evaluation Arguments
    parser.add_argument('--load-model', type=str, default=None,
                        help='Pretrained model name to load (state dict)')
    parser.add_argument('--save-model', type=str, default='model',
                        help='Pretrained model name to save (state dict)')
    parser.add_argument('--load-replay', type=str, default=None,
                        help='Load replay buffer from file')
    parser.add_argument('--save-replay', type=str, default=None,
                        help='Save replay buffer to file')



    #parser.add_argument('--evaluate', action='store_true',
    #                    help='Evaluate only')

    parser.add_argument('--evaluation_interval', type=int, default=100,
                        help='Frames for evaluation interval')
    parser.add_argument('--buffer-save-interval', type=int, default=100,
                        help='Frames for buffer save')
    parser.add_argument('--image-save-interval', type=int, default=10,
                        help='Frames for image save')





    # Optimization Arguments
    parser.add_argument('--lr', type=float, default=1e-4, metavar='η',
                        help='Learning rate')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args
