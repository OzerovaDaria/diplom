import gin.tf
import os
import copy
import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt
#from matplotlib.lines import Line2D
import pymetis

import sys
sys.path.append('./dte_stand')

from dte_stand.data_structures import HashWeights, Flow, InputData
from networkx.drawing.nx_agraph import write_dot
from typing import Optional, Iterable


DEFAULT_EDGE_ATTRIBUTES = {
    'increments': 1,
    'reductions': 1,
    'weight': 0.0,
    'current_bandwidth': 0.0
}

@gin.configurable
class Environment(object):
    def __init__(self,
                 current_topology,
                 hash_function,
                 env_type='Test',
                 traffic_profile='gravity_1',
                 routing='dxhash',
                 init_sample=0,
                 seed_init_weights=1,
                 min_weight=1.0,
                 max_weight=5.0,
                 weight_change=1.0,
                 weight_update='sum',
                 weigths_to_states=True,
                 link_traffic_to_states=True,
                 probs_to_states=False,
                 reward_magnitude='weights',
                 base_reward='min_max',
                 reward_computation='change',
                 base_dir='topologies',
                 graph_dir='dte_stand/algorithm/mate/graphs',
                 base_data_dir='data_examples',
                 topology='huawei.gml',
                 current_flows=[]):

        env_type = [env for env in env_type.split('+')]
        self.env_type = env_type
        self.traffic_profile = traffic_profile
        self.routing = routing
        self.topology = topology
        self.base_data_dir = base_data_dir

        self.num_sample = init_sample - 1
        self.seed_init_weights = seed_init_weights
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weight_change = weight_change
        self.weight_update = weight_update

        num_features = 0
        self.weigths_to_states = weigths_to_states
        if self.weigths_to_states:
            num_features += 1
        self.link_traffic_to_states = link_traffic_to_states
        if self.link_traffic_to_states:
            num_features += 1
        self.probs_to_states = probs_to_states
        if self.probs_to_states:
            num_features += 2
        self.num_features = num_features
        self.reward_magnitude = reward_magnitude
        self.base_reward = base_reward
        self.reward_computation = reward_computation
        self.base_dir = base_dir
        self.graph_dir = graph_dir
        self.current_topology = current_topology
        self.initialize_environment()
        self.get_weights()

        self.current_flows = current_flows
        self.hash_weights: Optional[HashWeights] = None
        self.hash_function = hash_function
        self.prev_edges = []

    def calculate_phi(self,  topology: nx.MultiDiGraph):
        number_of_edges = topology.number_of_edges()
        average_load: float = 0.0
        for edge in topology.edges(data=True):
            _, _, edge_data = edge
            average_load += float(edge_data['current_bandwidth']) / edge_data['bandwidth']
        average_load /= number_of_edges

        deviation: float = 0.0
        for edge in topology.edges(data=True):
            _, _, edge_data = edge
            deviation += pow(float(edge_data['current_bandwidth']) / edge_data['bandwidth'] - average_load, 2)
        return deviation / number_of_edges

    def get_current_flows(self, current_flows):
        self.current_flows = current_flows

    def _calculate_current_bandwidth(self, topology: nx.MultiDiGraph, flows: Iterable[Flow],
                                     hash_weights: HashWeights, horizon=None, num_sample=None) -> None:
        if hash_weights is None:
            return
        self.hash_function.run(topology, flows, hash_weights, False)

        '''
        edges = []
        free_edges = []
        for elem in topology.edges:
            if topology.edges[elem[0], elem[1], elem[2]]['current_bandwidth'] > 0:
                edges.append(elem)
            else:
                free_edges.append(elem)
        clrs = ['b', '#ff4500', '#00ff00', 'k']
        pos = nx.spiral_layout(topology)
        plt.figure(figsize=(15, 12))
        nx.draw_networkx_nodes(topology, pos, node_color='k', node_shape='.', node_size=40, alpha=0.8)
        nx.draw_networkx_labels(topology, pos, font_size=16, font_color='k', horizontalalignment='center',
                                verticalalignment='top',  ax=None, clip_on=True)
        nx.draw_networkx_edges(topology, pos, connectionstyle='arc3, rad = 0.1', edgelist=edges, width=0.6, alpha=0.5,
                                edge_color=clrs[3], style='solid', arrows=False)
                                
        '''

        '''
        nx.draw_networkx_edges(topology, pos, connectionstyle='arc3, rad = 0.1', edgelist=self.prev_edges,
                               edge_color=clrs[0], style='solid', alpha=1, arrows=True)
        nx.draw_networkx_edges(topology, pos, connectionstyle='arc3, rad = 0.1', edgelist=[x for x in edges if x not in self.prev_edges],
                               edge_color=clrs[1], style='solid', alpha=1, arrows=True)
        nx.draw_networkx_edges(topology, pos, connectionstyle='arc3, rad = 0.1', edgelist=[x for x in self.prev_edges if x not in edges],
                               edge_color=clrs[2], style='dashed', alpha=1, arrows=True)
        '''

        '''
        plt.axis('off')
        proxies = [Line2D([0, 1], [0, 1], color=clr, lw=2) for clr in clrs]
        plt.legend(proxies, ['unchanged traffic path', 'new traffic path', 'previously used traffic path', 'unused links'], fontsize=11)
        plt.savefig(self.graph_dir + '/iteration-' + str(num_sample) + '-horizon-' + '.png')
        self.prev_edges = edges
        plt.clf()
        '''

        self._get_link_traffic()

    def load_topology_object(self, current_topology=None):
        try:
            if current_topology:
                self.topology_object = current_topology
            else:
                nx_file = os.path.join(self.base_dir, self.topology)
                self.topology_object = nx.MultiDiGraph(nx.read_gml(nx_file, destringizer=int))

        except:
            print("Bad input of topology.gml")

    def initialize_environment(self, num_sample=None):
        if num_sample is not None:
            self.num_sample = num_sample
        else:
            self.num_sample += 1
        self.network = self.env_type[0]
        self.load_topology_object(self.current_topology)
        self.generate_graph()

    def next_sample(self):
        if len(self.env_type) > 1:
            self.initialize_environment()
        else:
            self.num_sample += 1
            self._reset_edge_attributes()

    def define_num_sample(self, num_sample):
        self.num_sample = num_sample - 1

    def reset(self, change_sample=False):
        if change_sample:
            self.next_sample()
        else:
            if self.seed_init_weights is None:
                self._define_init_weights()
            self._reset_edge_attributes()
        self.get_weights()
        self._get_HashWeights()
        self._calculate_current_bandwidth(self.G, self.current_flows, self.hash_weights)
        self.reward_measure = self.compute_reward_measure()
        self.set_target_measure()
        return self.get_state()

    def generate_graph(self):
        G = copy.deepcopy(self.topology_object)
        #for i, j, m in G.edges:
        #    print()
        self.n_nodes = G.number_of_nodes()
        self.n_links = G.number_of_edges()
        self._define_init_weights()
        idx = 0
        link_ids_dict = {}
        for i, j, m in G.edges:
            G[i][j][m]['label'] = G[i][j][m]['id']
            G[i][j][m]['id'] = idx
            G[i][j][m]['increments'] = 1
            G[i][j][m]['reductions'] = 1
            G[i][j][m]['weight'] = copy.deepcopy(self.init_weights[idx])
            link_ids_dict[idx] = (i, j, m)
            idx += 1
        self.G = G
        incoming_links, outcoming_links = self._generate_link_indices_and_adjacencies(self.G.nodes)
        #print("LINK IDS DICT", link_ids_dict)
        self.G.add_node('graph_data', link_ids_dict=link_ids_dict, incoming_links=incoming_links,
                        outcoming_links=outcoming_links)
        '''
        subgraphs = self.generate_subgraphs(2)
        for i in range(len(subgraphs)):
            print("SUBGRAPH", i, subgraphs[i])
            incoming_links, outcoming_links = self._generate_link_indices_and_adjacencies(subgraphs[i])
            print("INCOMING LINKS", incoming_links)
            print("OUTCOMING LINKS", outcoming_links)
            print()
            
            
            #commnet
            idx = 0
            link_ids_dict = {}
            for k, j, m in G.edges:
                if k in subgraphs[i] and j in subgraphs[i]:
                    link_ids_dict[idx] = (k, j, m)
                idx += 1
            print("LINK IDS DICT", link_ids_dict)
            ##commnet
            
            self.G.add_node('subgraph-' + str(i), link_ids_dict=link_ids_dict, incoming_links=incoming_links,
                        outcoming_links=outcoming_links)
        #print("MY GRAPH NODES", self.G.nodes['subgraph']['incoming_links'])
        '''

    def set_target_measure(self):
        self.target_reward_measure = copy.deepcopy(self.reward_measure)
        self.target_link_traffic = copy.deepcopy(self.link_traffic)
        self.get_weights()
        self.target_weights = copy.deepcopy(self.raw_weights)

    def get_weights(self):
        weights = [0.0] * self.n_links
        for i, j, m in self.G.edges(keys=True):
            weights[self.G.get_edge_data(i, j, m)['id']] = copy.deepcopy(self.G.get_edge_data(i, j, m)[
                "weight"])
        self.raw_weights = weights
        max_weight = self.max_weight * 3
        self.weights = [weight / max_weight for weight in weights]

    def get_state(self):
        state = []
        link_traffic = copy.deepcopy(self.link_traffic)
        weights = copy.deepcopy(self.weights)
        if self.link_traffic:
            state += link_traffic
        if self.weigths_to_states:
            state += weights
        if self.probs_to_states:
            state += self.p_in
            state += self.p_out
        return np.array(state, dtype=np.float32)

    def update_weights(self, link, action_value, get_state_back=False):
        i, j, m = link
        if self.weight_update == 'min_max':
            if action_value == 0:
                self.G[i][j][m]['weight'] = max(
                    self.G[i][j][m]['weight'] - self.weight_change, self.min_weight)
            elif action_value == 1:
                self.G[i][j][m]['weight'] = min(
                    self.G[i][j][m]['weight'] + self.weight_change, self.max_weight)
        else:
            if self.weight_update == 'increment_reduction':
                if action_value == 0:
                    self.G[i][j][m]['reductions'] += 1
                elif action_value == 1:
                    self.G[i][j][m]['increments'] += 1
                self.G[i][j][m]['weight'] = self.G[i][j][m]['increments'] / \
                    self.G[i][j][m]['reductions']
            elif self.weight_update == 'sum':
                if get_state_back:
                    self.G[i][j][m]['weight'] -= self.weight_change
                else:
                    self.G[i][j][m]['weight'] += self.weight_change

    def reinitialize_routing(self, routing):
        self.routing = routing
        self._get_link_traffic()

    def step(self, action, step_back=False, horizon=None, num_sample=None):
        #print("ACTION", action)
        #print("STEP LINK ID DICT", self.G.nodes()['subgraph-0']['link_ids_dict'])
        #link = self.G.nodes()['subgraph-0']['link_ids_dict'][action]
        link = self.G.nodes()['graph_data']['link_ids_dict'][action]
        self.update_weights(link, 0, step_back)
        self.get_weights()
        self._get_HashWeights()
        self._calculate_current_bandwidth(self.G, self.current_flows, self.hash_weights, horizon, num_sample)
        state = self.get_state()
        reward = self._compute_reward()
        return state, reward

    def step_back(self, action):
        state, reward = self.step(action, step_back=True)
        return state, reward

    def _define_init_weights(self):
        np.random.seed(seed=self.seed_init_weights)
        self.init_weights = np.random.randint(self.min_weight, self.max_weight + 1, self.n_links)
        np.random.seed(seed=None)

    def generate_subgraphs(self, num_of_subgraphs):
        adjacency_list = []
        for i in self.G.nodes():
            if np.fromiter(self.G.neighbors(i), int).size != 0:
                adjacency_list.append(np.fromiter(self.G.neighbors(i), int))
        #print("ADJENCY matrix", adjacency_list, len(adjacency_list), type(adjacency_list))
        n_cuts, membership = pymetis.part_graph(num_of_subgraphs, adjacency=adjacency_list)
        subgraphs = []
        for i in range(num_of_subgraphs):
            #print(i, np.argwhere(np.array(membership) == i).ravel())
            subgraphs.append(np.argwhere(np.array(membership) == i).ravel())
            subgraphs[i] = [str(x) for x in subgraphs[i]]
        return subgraphs

    def _generate_link_indices_and_adjacencies(self, subgraph_nodes=None):
        incoming_links = []
        outcoming_links = []
        #print("SUBGRAPH NODES", subgraph_nodes)
        for i in self.G.nodes():
            #print("I", i, type(i))
            if i != 'graph_data':
                if i in subgraph_nodes:
                    #print("I NEIGHBOURS node", i, np.fromiter(self.G.neighbors(i), int))
                    for j in self.G.neighbors(i):
                        if j in subgraph_nodes:
                            #print("J NEIGHBOURS of node", j, np.fromiter(self.G.neighbors(j), int))
                            incoming_link_id = self.G[i][j][0]['id']
                            for k in self.G.neighbors(j):
                                #if k in subgraph_nodes:
                                #print("K NEIGHBOURS of node", k, np.fromiter(self.G.neighbors(k), int))
                                outcoming_link_id = self.G[j][k][0]['id']

                                incoming_links.append(incoming_link_id)
                                outcoming_links.append(outcoming_link_id)
        #print("LINKS", incoming_links, outcoming_links)
        #print()
        return incoming_links, outcoming_links

    def _reset_edge_attributes(self, attributes=None):
        if attributes is None:
            attributes = list(DEFAULT_EDGE_ATTRIBUTES.keys())
        if type(attributes) != list:
            attributes = [attributes]
        for i, j, m in self.G.edges:
            for attribute in attributes:
                if attribute == 'weight':
                    self.G[i][j][m][attribute] = copy.deepcopy(
                        self.init_weights[self.G[i][j][m]['id']])
                else:
                    self.G[i][j][m][attribute] = copy.deepcopy(DEFAULT_EDGE_ATTRIBUTES[attribute])

    def _normalize_traffic(self):
        for i, j, m in self.G.edges:
            self.link_traffic[self.G[i][j][m]['id']] /= self.G[i][j][m]['bandwidth']

    def _get_link_traffic(self):
        link_traffic = [0] * self.n_links
        for i, j, m in self.G.edges:
            link_traffic[self.G[i][j][m]['id']] = self.G[i][j][m]['current_bandwidth']
        self.link_traffic = link_traffic
        self._normalize_traffic()
        self.mean_traffic = np.mean(link_traffic)
        self.get_weights()

    def compute_reward_measure(self, measure=None):
        if measure is None:
            if self.reward_magnitude == 'link_traffic':
                measure = self.link_traffic
            elif self.reward_magnitude == 'weights':
                measure = self.raw_weights
        if self.base_reward == 'mean_times_std':
            return np.mean(measure) * np.std(measure)
        elif self.base_reward == 'mean':
            return np.mean(measure)
        elif self.base_reward == 'std':
            return np.std(measure)
        elif self.base_reward == 'diff_min_max':
            return np.max(measure) - np.min(measure)
        elif self.base_reward == 'min_max':
            return np.max(measure)

    def _compute_reward(self, current_reward_measure=None):
        if current_reward_measure is None:
            current_reward_measure = self.compute_reward_measure()
        if self.reward_computation == 'value':
            reward = - current_reward_measure
        elif self.reward_computation == 'change':
            reward = self.reward_measure - current_reward_measure
        self.reward_measure = current_reward_measure
        return reward

    def _get_HashWeights(self):
        hash_weights = HashWeights()
        topo_nodes = self.G.nodes()
        for start_node in topo_nodes:
            for end_node in topo_nodes:
                if start_node == end_node:
                    continue
                try:
                    node_edges = list(self.G.edges(nbunch=start_node, keys=True))
                except KeyError:
                    print("node was removed from topology")
                    continue
                for edge in node_edges:
                    edge_start, edge_end, edge_index = edge
                    edge_weight = (self.G.get_edge_data(edge_start, edge_end, edge_index)[
                        "weight"])
                    hash_weights.put(edge_start, end_node, edge_end, edge_index, edge_weight)
        self.hash_weights = hash_weights