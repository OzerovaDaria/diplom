import gin.tf
import numpy as np
import tensorflow as tf
from tensorflow import keras


@gin.configurable
class Actor(keras.Sequential):
    def __init__(self,
                 graph,
                 num_actions=1,
                 num_features=2,
                 link_state_size=16,
                 aggregation='min_max',
                 first_hidden_layer_size=128,
                 dropout_rate=0.15,
                 final_hidden_layer_size=64,
                 message_iterations=8,
                 activation_fn='tanh',
                 final_activation_fn='linear'):

        super(Actor, self).__init__()
        # HYPERPARAMETERS
        self.num_actions = num_actions
        self.num_features = num_features
        self.n_links = graph.number_of_edges()
        self.link_state_size = link_state_size
        self.message_hidden_layer_size = final_hidden_layer_size
        self.aggregation = aggregation
        self.message_iterations = message_iterations

        # FIXED INPUTS
        #print("GRAPH DATA", graph.nodes()['graph_data'])
        #self.incoming_links = graph.nodes()['subgraph-0']['incoming_links']
        #self.outcoming_links = graph.nodes()['subgraph-0']['outcoming_links']
        self.incoming_links = graph.nodes()['graph_data']['incoming_links']
        self.outcoming_links = graph.nodes()['graph_data']['outcoming_links']
        #print("INCOMING LINKS", self.incoming_links)
        #print("OUTCOMING LINKS", self.outcoming_links)

        # NEURAL NETWORKS
        self.hidden_layer_initializer = tf.keras.initializers.Orthogonal(gain=np.sqrt(2))
        self.final_layer_initializer = tf.keras.initializers.Orthogonal(gain=0.01)
        self.kernel_regularizer = None
        self.activation_fn = activation_fn
        self.final_hidden_layer_size = final_hidden_layer_size
        self.first_hidden_layer_size = first_hidden_layer_size
        self.dropout_rate = dropout_rate
        self.final_activation_fn = final_activation_fn
        self.define_network()

    def define_network(self):
        self.create_message = keras.models.Sequential(name='create_message')
        self.create_message.add(keras.layers.Dense(self.message_hidden_layer_size,
                                                   kernel_initializer=self.hidden_layer_initializer,
                                                   activation=self.activation_fn))
        self.create_message.add(keras.layers.Dense(self.link_state_size,
                                                   kernel_initializer=self.hidden_layer_initializer,
                                                   activation=self.activation_fn))

        # link update
        self.link_update = keras.models.Sequential(name='link_update')
        self.link_update.add(keras.layers.Dense(self.first_hidden_layer_size,
                                                kernel_initializer=self.hidden_layer_initializer,
                                                activation=self.activation_fn))
        self.link_update.add(keras.layers.Dense(self.final_hidden_layer_size,
                                                kernel_initializer=self.hidden_layer_initializer,
                                                activation=self.activation_fn))
        self.link_update.add(keras.layers.Dense(self.link_state_size,
                                                kernel_initializer=self.hidden_layer_initializer,
                                                activation=self.activation_fn))
        self.readout = keras.models.Sequential(name='readout')
        self.readout.add(
            keras.layers.Dense(self.first_hidden_layer_size, kernel_initializer=self.hidden_layer_initializer,
                               kernel_regularizer=self.kernel_regularizer, activation=self.activation_fn))
        self.readout.add(keras.layers.Dropout(self.dropout_rate))
        self.readout.add(
            keras.layers.Dense(self.final_hidden_layer_size, kernel_initializer=self.hidden_layer_initializer,
                               kernel_regularizer=self.kernel_regularizer, activation=self.activation_fn))
        self.readout.add(keras.layers.Dropout(self.dropout_rate))
        self.readout.add(keras.layers.Dense(self.num_actions, kernel_initializer=self.final_layer_initializer,
                                            kernel_regularizer=self.kernel_regularizer,
                                            activation=self.final_activation_fn))

    def build(self, input_shape=None):
        self.create_message.build(input_shape=[None, 2 * self.link_state_size])
        if self.aggregation == 'sum':
            self.link_update.build(input_shape=[None, 2 * self.link_state_size])
        elif self.aggregation == 'min_max':
            self.link_update.build(input_shape=[None, 3 * self.link_state_size])
        self.readout.build(input_shape=[None, self.link_state_size])
        self.built = True

    @tf.function
    def message_passing(self, input):
        input_tensor = tf.convert_to_tensor(input)
        link_states = tf.reshape(input_tensor, [self.num_features, self.n_links])
        link_states = tf.transpose(link_states)
        padding = [[0, 0], [0, self.link_state_size - self.num_features]]
        link_states = tf.pad(link_states, padding)
        for _ in range(self.message_iterations):
            incoming_link_states = tf.gather(link_states, self.incoming_links)
            outcoming_link_states = tf.gather(link_states, self.outcoming_links)
            message_inputs = tf.cast(tf.concat([incoming_link_states, outcoming_link_states], axis=1), tf.float32)
            messages = self.create_message(message_inputs)
            aggregated_messages = self.message_aggregation(messages)
            link_update_input = tf.cast(tf.concat([link_states, aggregated_messages], axis=1), tf.float32)
            link_states = self.link_update(link_update_input)
        return link_states

    @tf.function
    def message_aggregation(self, messages):
        if self.aggregation == 'sum':
            aggregated_messages = tf.math.unsorted_segment_sum(messages, self.outcoming_links,
                                                               num_segments=self.n_links)
        elif self.aggregation == 'min_max':
            agg_max = tf.math.unsorted_segment_max(messages, self.outcoming_links, num_segments=self.n_links)
            agg_min = tf.math.unsorted_segment_min(messages, self.outcoming_links, num_segments=self.n_links)
            aggregated_messages = tf.concat([agg_max, agg_min], axis=1)
        return aggregated_messages

    @tf.function
    def call(self, input):
        link_states = self.message_passing(input)
        #print("LINK STATES", link_states)
        policy = self.readout(link_states)
        policy = tf.reshape(policy, [-1])
        #print("POLICY", policy)
        return policy