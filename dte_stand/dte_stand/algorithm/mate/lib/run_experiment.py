import logging
import os

import gin.tf
import tensorflow as tf

from dte_stand.algorithm.mate.agents.ppo_agent import PPOAgent
from dte_stand.algorithm.mate.environment.environment import Environment

@gin.configurable
class Runner(object):
    def __init__(self,
                 topology_object,
                 hash_function,
                 reload_model,
                 algorithm='PPO',
                 # model_dir='checkpoints/training/gravity_1/PPO_agg_period100/clip0.2/gamma0.95/episode',
                 model_dir='dte_stand/algorithm/mate/checkpoints/training/Test-gravity_1-dxhash/batch25-lr0.0003-epsilon0.1-gae0.9-clip0.2-gamma0.95-period5-epoch3/size16-iters8-min_max-nnsize64-drop0.15-tanh//episode2',
                 only_eval=False,
                 base_dir='dte_stand/algorithm/mate/logs',
                 checkpoint_base_dir='dte_stand/algorithm/mate/checkpoints',
                 save_checkpoints=True):

        self.save_checkpoints = save_checkpoints
        self.hash_function = hash_function
        env = Environment(topology_object, self.hash_function)
        agent = PPOAgent
        if algorithm == 'PPO':
            self.agent = agent(env)
        else:
            assert False, 'RL Algorithm %s is not implemented' % algorithm
        self.base_dir = base_dir
        self.checkpoint_base_dir = checkpoint_base_dir
        self.only_eval = only_eval
        if reload_model or self.only_eval:
            self.agent.load_saved_model(model_dir, only_eval)
        self.set_logs_and_checkpoints()

    def run_experiment(self, topology, current_flows, iteration):
        hash_weights, phi_dict = self.agent.train_and_evaluate(topology, current_flows, self.hash_function, iteration)
        #return hash_weights, phi_dict
        print("PHI DCT RETURNED", phi_dict)
        return hash_weights, phi_dict

    def set_logs_and_checkpoints(self):
        experiment_identifier = self.agent.set_experiment_identifier(self.only_eval)
        writer_dir = os.path.join(self.base_dir, experiment_identifier)
        if not os.path.exists(writer_dir):
            os.makedirs(writer_dir)
        checkpoint_dir = os.path.join(self.checkpoint_base_dir, experiment_identifier)
        if self.save_checkpoints and (not os.path.exists(checkpoint_dir)):
            os.makedirs(checkpoint_dir)
        self.agent.set_writer_and_checkpoint_dir(writer_dir, checkpoint_dir)
        f = open(os.path.join(writer_dir, 'out.log'), 'w+')
        f.close()
        fh = logging.FileHandler(os.path.join(writer_dir, 'out.log'))
        fh.setLevel(logging.DEBUG)
        tf.get_logger().addHandler(fh)