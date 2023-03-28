import copy
import csv
import os
import random
import gin.tf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from dte_stand.phi_calculator import PhiCalculator
import dte_stand.algorithm.mate.utils.tf_logs as tf_logs
from dte_stand.algorithm.mate.environment.environment import Environment
from dte_stand.algorithm.mate.lib.actor import Actor
from dte_stand.algorithm.mate.lib.critic import Critic
from dte_stand.config import MateActions
from dte_stand.history import HistoryTracker


@gin.configurable
class PPOAgent(object):
    """An implementation of a GNN-based PPO Agent"""

    def __init__(self,
                 env,
                 action_config: MateActions,
                 phi_func,
                 message_iterations,
                 eval_env_type=['Test'],
                 plot_period=1000,
                 num_eval_samples=3,
                 clip_param=0.25,
                 critic_loss_factor=0.5,
                 entropy_loss_factor=0.001,
                 normalize_advantages=True,
                 max_grad_norm=1.0,
                 gamma=0.99,
                 gae_lambda=0.95,
                 horizon=100,
                 batch_size=25,
                 epochs=3,
                 last_training_sample=1,
                 eval_period=150,
                 max_evals=5,
                 select_max_action=False,
                 optimizer=tf.keras.optimizers.legacy.Adam(  # for tensorflow<2.11 remove legacy!
                     learning_rate=0.0003,
                     beta_1=0.9,
                     epsilon=0.00001),
                 change_traffic=True,
                 change_traffic_period=1,
                 base_dir='logs',
                 checkpoint_dir='checkpoints',
                 save_checkpoints=True,
                 greedy_eplison=1):

        self.phi = phi_func
        self.env = env
        self.action_config = action_config
        self.eval_env_type = eval_env_type
        self.num_eval_samples = num_eval_samples
        self.clip_param = clip_param

        self.actor = None
        self.message_iterations = message_iterations
        self._get_actor_critic_functions()
        self.optimizer = optimizer
        self.critic_loss_factor = critic_loss_factor
        self.entropy_loss_factor = entropy_loss_factor
        self.normalize_advantages = normalize_advantages
        self.max_grad_norm = max_grad_norm

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.given_horizon = horizon
        self.define_horizon()
        self.epochs = epochs
        self.batch_size = batch_size
        self.last_training_sample = last_training_sample
        self.eval_period = eval_period
        self.max_evals = max_evals
        self.select_max_action = select_max_action
        self.change_traffic = change_traffic
        self.change_traffic_period = change_traffic_period
        self.eval_step = 0
        self.eval_episode = 0
        self.base_dir = base_dir
        self.checkpoint_dir = checkpoint_dir
        self.save_checkpoints = save_checkpoints
        self.reload_model = False
        self.change_sample = False
        self.eps = greedy_eplison
        self.plot_period = plot_period
        self.tracker = HistoryTracker()
        self.set_experiment_identifier(False)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def _get_actor_critic_functions(self):
        self.actions = {}
        self.num_actions = 0

        if self.action_config.addition.action:
            self.actions[self.num_actions] = '+'
            self.num_actions += 1
        if self.action_config.subtraction.action:
            self.actions[self.num_actions] = '-'
            self.num_actions += 1
        if self.action_config.multiplication.action:
            self.actions[self.num_actions] = '*'
            self.num_actions += 1
        if self.action_config.division.action:
            self.actions[self.num_actions] = '/'
            self.num_actions += 1
        if self.action_config.zero.action:
            self.actions[self.num_actions] = '0'
            self.num_actions += 1

        self.actor = Actor(self.env.G, num_actions=self.num_actions, num_features=self.env.num_features,
                           message_iterations=self.message_iterations)
        self.actor.build()
        self.critic = Critic(self.env.G, num_features=self.env.num_features, message_iterations=self.message_iterations)
        self.critic.build()

    def define_horizon(self):
        if self.given_horizon is not None:
            self.horizon = self.given_horizon
        elif self.env.topology == 'test.gml':
            self.horizon = 30
        elif self.env.topology == 'huawei.gml':
            self.horizon = 40
        else:
            self.horizon = 50

    def reset_env(self):
        self.env.reset(change_sample=self.change_sample)
        if self.change_sample and len(self.env.env_type) > 1:
            actor_model = copy.deepcopy(self.actor.trainable_variables)
            critic_model = copy.deepcopy(self.critic.trainable_variables)
            self._get_actor_critic_functions()
            self.load_model(actor_model, critic_model)
            self.define_horizon()
        self.change_sample = False

    def gae_estimation(self, rewards, values, last_value):
        last_gae_lambda = 0
        advantages = np.zeros_like(values, dtype=np.float32)
        for i in reversed(range(self.horizon)):
            if i == self.horizon - 1:
                next_value = last_value
            else:
                next_value = values[i + 1]
            delta = rewards[i] + self.gamma * next_value - values[i]
            advantages[i] = last_gae_lambda = delta + self.gamma * self.gae_lambda * last_gae_lambda
        returns = values + advantages
        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return returns, advantages

    def run_episode(self):
        self.reset_env()
        state = self.env.get_state()
        states = np.zeros((self.horizon, self.env.n_links *
                           self.actor.num_features), dtype=np.float32)
        actions = np.zeros(self.horizon, dtype=np.float32)
        rewards = np.zeros(self.horizon, dtype=np.float32)
        log_probs = np.zeros(self.horizon, dtype=np.float32)
        values = np.zeros(self.horizon, dtype=np.float32)
        weights = []
        probabilities = []

        for t in range(self.horizon):
            action, log_prob, raw_probs = self.act(state)
            value = self.run_critic(state)
            action_numpy = action.numpy()
            action_type = self.actions[action_numpy % self.num_actions]
            act_val = action_numpy // self.num_actions
            next_state, reward = self.env.step(act_val, action_type, last=(True if t == self.horizon - 1 else False))

            states[t] = state
            actions[t] = action
            rewards[t] = reward
            log_probs[t] = log_prob
            values[t] = value.numpy()[0]
            state = next_state
        value = self.run_critic(state)
        last_value = value.numpy()[0]
        self.tracker.add_value('actions', actions)
        self.tracker.add_value('rewards', rewards)
        return states, actions, rewards, log_probs, values, last_value

    def run_update(self, states, actions, returns, advantages, log_probs):
        actor_losses, critic_losses, losses = [], [], []
        inds = np.arange(self.horizon)
        for _ in range(self.epochs):
            np.random.shuffle(inds)
            for start in range(0, self.horizon, self.batch_size):
                end = start + self.batch_size
                minibatch_ind = inds[start:end]
                actor_loss, critic_loss, loss, grads = self.compute_losses_and_grads(states[minibatch_ind],
                                                                                     actions[minibatch_ind],
                                                                                     returns[minibatch_ind],
                                                                                     advantages[minibatch_ind],
                                                                                     log_probs[minibatch_ind])
                self.apply_grads(grads)
                actor_losses.append(actor_loss.numpy())
                critic_losses.append(critic_loss.numpy())
                losses.append(loss.numpy())
        return actor_losses, critic_losses, losses

    def train_and_evaluate(self, topology, current_flows, hash_function):
        training_episode = -1
        self.env.get_current_flows(current_flows)
        while not self.change_sample:
            training_episode += 1
            print('Episode ', training_episode, '...')
            states, actions, rewards, log_probs, values, last_value = self.run_episode()
            returns, advantages = self.gae_estimation(rewards, values, last_value)

            # actor_losses, critic_losses, losses = self.run_update(states, actions, returns,
            #                                                       advantages, log_probs)
            # tf_logs.training_episode_logs(self.writer, self.env, training_episode, states, rewards, losses,
            #                               actor_losses, critic_losses)

            if (training_episode + 1) % self.eval_period == 0:
                # self.training_eval(topology, current_flows, hash_function)
                if self.save_checkpoints:
                    self.actor._set_inputs(states[0])
                    self.critic._set_inputs(states[0])
                    self.save_model(self.checkpoint_dir)
                if self.change_traffic and self.eval_episode % self.change_traffic_period == 0:
                    self.change_sample = True

            print(f'phi = {self.phi(self.env.G)}')
            PhiCalculator.end_episode()
            self.tracker.end_iteration()
            self.env.end_iteration()

            if training_episode > 0 and ((training_episode + 1) % self.plot_period == 0):
                PhiCalculator.plot_result()
        self.env.get_hash_weights()
        return self.env.hash_weights

    def generate_eval_env(self, current_flows, current_topology, hash_function):
        self.eval_envs = {}
        for eval_env_type in self.eval_env_type:
            self.eval_envs[eval_env_type] = Environment(env_type=eval_env_type,
                                                        traffic_profile=self.env.traffic_profile,
                                                        routing=self.env.routing,
                                                        current_flows=current_flows,
                                                        current_topology=current_topology,
                                                        hash_function=hash_function,
                                                        action_config=self.action_config)

    def generate_eval_actor_critic_functions(self):
        self.eval_actor = {}
        self.eval_critic = {}
        for eval_env_type in self.eval_env_type:
            self.eval_actor[eval_env_type] = Actor(
                self.eval_envs[eval_env_type].G, num_features=self.env.num_features)
            self.eval_actor[eval_env_type].build()
            self.eval_critic[eval_env_type] = Critic(self.eval_envs[eval_env_type].G,
                                                     num_features=self.env.num_features)
            self.eval_critic[eval_env_type].build()

    def update_eval_actor_critic_functions(self):
        for eval_env_type in self.eval_env_type:
            for w_model, w_eval_actor in zip(self.actor.trainable_variables,
                                             self.eval_actor[eval_env_type].trainable_variables):
                w_eval_actor.assign(w_model)
            for w_model, w_eval_critic in zip(self.critic.trainable_variables,
                                              self.eval_critic[eval_env_type].trainable_variables):
                w_eval_critic.assign(w_model)

    def training_eval(self, current_topology, current_flows, hash_function):
        if self.eval_episode == 0:
            self.generate_eval_env(current_flows, current_topology, hash_function)
            self.generate_eval_actor_critic_functions()
        self.update_eval_actor_critic_functions()
        for eval_env_type in self.eval_env_type:
            self.eval_envs[eval_env_type].define_num_sample(100)
            total_min_max = []
            mini_eval_episode = self.eval_episode * self.num_eval_samples
            for j in range(self.num_eval_samples):
                self.eval_envs[eval_env_type].reset(change_sample=True)
                state = self.eval_envs[eval_env_type].get_state()
                if self.eval_envs[eval_env_type].link_traffic_to_states:
                    max_link_utilization = [np.max(state[:self.eval_envs[eval_env_type].n_links])]
                probs, values = [], []
                for i in range(self.horizon):
                    self.eval_step += 1
                    action, log_prob = self.eval_act(self.eval_actor[eval_env_type], state,
                                                     select_max=self.select_max_action)

                    value = self.eval_critic[eval_env_type](state)
                    next_state, reward = self.eval_envs[eval_env_type].step(action.numpy(), 0, i, j)
                    probs.append(np.exp(log_prob))
                    values.append(value.numpy()[0])
                    state = next_state
                    if self.eval_envs[eval_env_type].link_traffic_to_states:
                        max_link_utilization.append(
                            np.max(state[:self.eval_envs[eval_env_type].n_links]))
                if self.env.link_traffic_to_states:
                    total_min_max.append(np.min(max_link_utilization))
                    # tf_logs.eval_final_log(self.writer, mini_eval_episode, max_link_utilization, eval_env_type)
                mini_eval_episode += 1
            # tf_logs.eval_top_log(self.writer, self.eval_episode, total_min_max, eval_env_type)
        self.eval_episode += 1

    @tf.function
    def compute_actor_loss(self, new_log_probs, old_log_probs, advantages):
        ratio = tf.exp(new_log_probs - old_log_probs)
        pg_loss_1 = - advantages * ratio
        pg_loss_2 = - advantages * \
            tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        actor_loss = tf.reduce_mean(tf.maximum(pg_loss_1, pg_loss_2))
        return actor_loss

    @tf.function
    def get_new_log_prob_and_entropy(self, state, action):
        logits = self.actor(state, training=True)
        probs = tfp.distributions.Categorical(logits=logits)
        return (probs.log_prob(action), probs.entropy())

    @tf.function
    def compute_losses_and_grads(self, states, actions, returns, advantages, old_log_probs):
        with tf.GradientTape(persistent=True) as tape:
            new_log_probs, entropy = tf.map_fn(lambda x: self.get_new_log_prob_and_entropy(x[0], x[1]),
                                               (states, actions), fn_output_signature=(tf.float32, tf.float32))
            values = tf.map_fn(lambda x: self.critic(x, training=True),
                               states, fn_output_signature=tf.float32)
            values = tf.reshape(values, [-1])
            critic_loss = tf.reduce_mean(tf.square(returns - values))
            entropy_loss = tf.reduce_mean(entropy)
            actor_loss = self.compute_actor_loss(new_log_probs, old_log_probs, advantages)
            loss = actor_loss - self.entropy_loss_factor * entropy_loss + self.critic_loss_factor * critic_loss
        grads = tape.gradient(loss, self.actor.trainable_variables +
                              self.critic.trainable_variables)
        if self.max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        return actor_loss, critic_loss, loss, grads

    def apply_grads(self, grads):
        self.optimizer.apply_gradients(
            zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))

    @tf.function
    def act(self, state, select_max=False):
        logits = self.actor(state)
        probs = tfp.distributions.Categorical(logits=logits)
        probabilities = [probs.prob(x) for x in range(len(logits))]

        if random.uniform(0, 1) > self.eps:
            action = tf.argmax(logits)
        else:
            action = probs.sample()
        return action, probs.log_prob(action), probabilities

    @tf.function
    def eval_act(self, actor, state, select_max=False):
        logits = actor(state)
        probs = tfp.distributions.Categorical(logits=logits)
        if select_max:
            action = tf.argmax(logits)
        else:
            action = probs.sample()
        return action, probs.log_prob(action)

    @tf.function
    def run_critic(self, state):
        return self.critic(state)

    def save_model(self, checkpoint_dir):
        self.actor.save(checkpoint_dir + '/actor')
        self.critic.save(checkpoint_dir + '/critic')

    def load_model(self, actor_model, critic_model):
        for w_model, w_actor in zip(actor_model,
                                    self.actor.trainable_variables):
            w_actor.assign(w_model)
        for w_model, w_critic in zip(critic_model,
                                     self.critic.trainable_variables):
            w_critic.assign(w_model)

    def load_saved_model(self, model_dir, only_eval):
        model = keras.models.load_model(model_dir + '/actor')
        for w_model, w_actor in zip(model.trainable_variables,
                                    self.actor.trainable_variables):
            w_actor.assign(w_model)
        if not only_eval:
            model = keras.models.load_model(model_dir + '/critic')
            for w_model, w_critic in zip(model.trainable_variables,
                                         self.critic.trainable_variables):
                w_critic.assign(w_model)
        self.model_dir = model_dir
        self.reload_model = True

    def write_eval_results(self, step, value):
        csv_dir = os.path.join('./notebooks/logs', self.experiment_identifier)
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        with open(csv_dir + '/results.csv', "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow([step, value])

    def set_experiment_identifier(self, only_eval):
        self.only_eval = only_eval
        mode = 'eval' if only_eval else 'training'

        if mode == 'training':
            # PPOAGENT
            batch = 'batch' + str(self.batch_size)
            gae_lambda = 'gae' + str(self.gae_lambda)
            lr = 'lr' + str(self.optimizer.get_config()['learning_rate'])
            epsilon = 'epsilon' + str(self.optimizer.epsilon)
            clip = 'clip' + str(self.clip_param)
            gamma = 'gamma' + str(self.gamma)
            episodes = 'episodes' + str(self.eval_period)
            horizon = 'horizons' + str(self.horizon)
            epoch = 'epoch' + str(self.epochs)
            greedy_eps = 'greedyeps' + str(self.eps)
            agent_folder = '-'.join((batch, lr, epsilon, gae_lambda, clip, gamma,
                                     episodes, horizon, epoch, greedy_eps))

            # ACTOR-CRITIC-ENV
            state_size = 'size' + str(self.actor.link_state_size)
            iters = 'iters' + str(self.actor.message_iterations)
            aggregation = self.actor.aggregation
            nn_size = 'nnsize' + str(self.actor.final_hidden_layer_size)
            dropout = 'drop' + str(self.actor.dropout_rate)
            activation = self.actor.activation_fn
            base_reward = self.env.base_reward
            reward_comp = self.env.reward_computation
            function_folder = '-'.join((state_size, iters, aggregation, nn_size, dropout, activation,
                                        base_reward, reward_comp))

            self.experiment_identifier = os.path.join(agent_folder, function_folder)

        else:
            model_dir = self.model_dir

            network = '+'.join([str(elem) for elem in self.env.env_type])
            traffic_profile = self.env.traffic_profile
            routing = self.env.routing
            eval_env_folder = ('-').join([network, traffic_profile, routing])

            # RELOADED MODEL
            env_folder = os.path.join(model_dir.split('/')[3])
            agent_folder = os.path.join(model_dir.split('/')[4])
            function_folder = os.path.join(model_dir.split('/')[5])
            episode = os.path.join(model_dir.split('/')[6])

            self.experiment_identifier = os.path.join(mode, eval_env_folder, env_folder, agent_folder, function_folder,
                                                      episode)
        return self.experiment_identifier
