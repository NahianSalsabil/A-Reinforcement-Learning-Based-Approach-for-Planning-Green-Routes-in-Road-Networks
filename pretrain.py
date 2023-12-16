import tensorflow as tf
import networkx as nx
import numpy as np
import keras
from keras.layers import Dense
from keras.optimizers import Adam
import xmltodict
import pandas as pd
import networkx as nx
import math
import random
import matplotlib.pyplot as plt
from queue import PriorityQueue


def calculate_initial_compass_bearing(pointA, pointB):
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def name_angle(angle):
    if angle <= 22.5 or angle > 337.5:
        return "NORTH"
    elif angle > 22.5 and angle <= 67.5:
        return "NORTHEAST"
    elif angle > 67.5 and angle <= 112.5:
        return "EAST"
    elif angle > 112.5 and angle <= 157.5:
        return "SOUTHEAST"
    elif angle > 157.5 and angle <= 202.5:
        return "SOUTH"
    elif angle > 202.5 and angle <= 247.5:
        return "SOUTHWEST"
    elif angle > 247.5 and angle <= 292.5:
        return "WEST"
    elif angle > 292.5 and angle <= 337.5:
        return "NORTHWEST"

def id_angle(angle):
    if angle <= 22.5 or angle > 337.5:
        return 0
    elif angle > 22.5 and angle <= 67.5:
        return 1
    elif angle > 67.5 and angle <= 112.5:
        return 2
    elif angle > 112.5 and angle <= 157.5:
        return 3
    elif angle > 157.5 and angle <= 202.5:
        return 4
    elif angle > 202.5 and angle <= 247.5:
        return 5
    elif angle > 247.5 and angle <= 292.5:
        return 6
    elif angle > 292.5 and angle <= 337.5:
        return 7
    
def nam_to_id(name):
    if name=="NORTH":
        return 0
    elif name=="NORTHEAST":
        return 1
    elif name=="EAST":
        return 2
    elif name=="SOUTHEAST":
        return 3
    elif name=="SOUTH":
        return 4
    elif name=="SOUTHWEST":
        return 5
    elif name=="WEST":
        return 6
    elif name=="NORTHWEST":
        return 7

def id_to_name(id):
    if id==0:
        return "NORTH"
    elif id==1:
        return "NORTHEAST"
    elif id==2:
        return "EAST"
    elif id==3:
        return "SOUTHEAST"
    elif id==4:
        return "SOUTH"
    elif id==5:
        return "SOUTHWEST"
    elif id==6:
        return "WEST"
    elif id==7:
        return "NORTHWEST"

def retriveGraph(file):
    with open(file, 'r') as f:
        xml_data=f.read()
        xml_to_dict=xmltodict.parse(xml_data)
        edges=[]
    nodes={}
    neighbor={}
    for edge in xml_to_dict['graphml']['graph']['edge']:
        src=edge['@source']

        dest=edge['@target']

        if(src==dest):
            continue
        new_node1={}
        if(edge['data'][2]['#text'][12:-1].split(',')[0].strip().split(' ')[0]!='ECTION'):
            new_node1['lat']=float(edge['data'][2]['#text'][12:-1].split(',')[0].strip().split(' ')[0])
            new_node1['lng']=float(edge['data'][2]['#text'][12:-1].split(',')[0].strip().split(' ')[1])
            nodes[src]=new_node1
            new_node2={}
            new_node2['lat']=float(edge['data'][2]['#text'][12:-1].split(',')[-1].strip().split(' ')[0])
            new_node2['lng']=float(edge['data'][2]['#text'][12:-1].split(',')[-1].strip().split(' ')[1])
            nodes[dest]=new_node2
            
            if src not in neighbor.keys():
                neighbor[src]=[]
            if dest not in neighbor.keys():
                neighbor[dest]=[]
            if dest not in neighbor[src]:
                neighbor[src].append(dest)
            if src not in neighbor[dest]:
                neighbor[dest].append(src)
            new_edge={}
            new_edge['u']=src
            new_edge['v']=dest
            
            for data in edge['data']:
                if(data['@key']!='e_geom'and data['@key']!='e_geom_wgs'):
                    new_edge[data['@key']]=data['#text']

            new_edge['action']=id_angle(calculate_initial_compass_bearing((nodes[src]['lat'],nodes[src]['lng']),(nodes[dest]['lat'],nodes[dest]['lng'])))
            edges.append(new_edge)
    return nodes,edges,neighbor
file_path='kumpula.graphml'
nodes,edges,neighbor=retriveGraph(file_path)
# Define the neural network for the policy

G = pd.read_pickle("full_graph.gpickle")

def get_next_state(graph,node,action_chosen):
    next_state=None
    next_states = list(graph.neighbors(node))
    actions=[]
    for n in  next_states:
        action=id_angle(calculate_initial_compass_bearing((nodes[node]['lat'],nodes[node]['lng']),(nodes[n]['lat'],nodes[n]['lng']))) 
        if(action==action_chosen):
            actions.append(n)
    next_state=random.choice(actions)
    reward=graph[node][next_state]['aqi']+graph[node][next_state]['e_g']
    # print('reward: ',reward,node,next_state,graph[node][next_state]['aqi'],graph[node][next_state]['e_g'])
    # if(next==target):done=1
    # else: done=0
    return next_state,reward

def state_embedding(state_idx):
    curr=node_embeddings[state_idx[0]]
    targ=node_embeddings[state_idx[1]]
    state=np.expand_dims(np.concatenate((np.asarray(curr), np.asarray(targ) - np.asarray(curr))), axis=0)
    return state


class Env(object):
    def __init__(self):
        self.path = []
        self.path_no_relations = []
        self.path_relations = []
        self.coords_path = []
        self.curr_rewards = 0.0
        self.curr_prob = 1.0
        self.edge_rewards = {}
        self.edge_transition_probs = {}
        self.die = 0  # record how many times does the agent choose an invalid path
        self.last_visited = {}
        self.visited = {}
        self.done = 0
        self.steps = 0
        self.path_length = 0.0

    def copy(self):
        o = Env.__new__(Env)
        o.path = list(self.path)
        o.path_relations = list(self.path_relations)
        o.path_no_relations = list(self.path_no_relations)
        o.coords_path = list(self.coords_path)
        o.curr_rewards = self.curr_rewards
        o.curr_prob = self.curr_prob
        o.edge_rewards = {k: dict(v.items()) for k, v in self.edge_rewards.items()}
        o.edge_transition_probs = {k: dict(v.items()) for k, v in self.edge_transition_probs.items()}
        o.die = self.die  # record how many times does the agent choose an invalid path
        o.last_visited = {k: tuple(v) for k, v in self.last_visited.items()}
        o.visited = {k: set(v) for k, v in self.visited.items()}
        o.done = self.done
        o.steps = self.steps
        o.path_length = self.path_length
        return o
    
    def __lt__(self, other):
        # Define the comparison logic between instances of Env
        return self.curr_prob < other.curr_prob

    # def __lt__(self, other):
    #     # Define the comparison logic between instances of Env
    #     return self.curr_prob < other.curr_prob

    def interact(self, state, action, rollout, prob):
        done = 0  # Whether the episode has finished
        cur = state[0]
        dest = state[1]
        next_node,edge_reward=get_next_state(G,cur,action)
        if(next_node==None):
            print('the action is invalid ,no next state from here')
        else:
            self.path.append(str(action)+'->'+next_node) #keno use hocce buji nai
            self.path_no_relations.append(next_node) #keno use hocce buji nai
            self.coords_path.append((nodes[next_node]['lat'],nodes[next_node]['lng'])) #keno use hocce buji nai

            # add action taken to dictionary of state-action pairs
            self.last_visited[cur] = (action,next_node,len(self.path) - 1) #3rd param buji nai
            if cur in self.visited:
                self.visited[cur].add(action)
            else:
                self.visited[cur] = {action}

            self.die = 0
            if next_node == dest and not rollout:
                done = 1
                self.done = 1


        # save above reward data for each node visited for future rewards
        if cur in self.edge_rewards:
            if next_node not in self.edge_rewards[cur]:
                self.edge_rewards[cur][next_node] = (edge_reward)
        else:
            self.edge_rewards[cur] = {next_node:edge_reward}

        # save probabilities for each node taken
        if cur in self.edge_transition_probs:
            if next_node not in self.edge_transition_probs[cur]:
                self.edge_transition_probs[cur][next_node] = prob
        else:
            self.edge_transition_probs[cur] = {next_node: prob}

        # print('in interact methdo: ')
        # print('cur,nest: ',cur,next_node)
        # print('probs: ',self.edge_transition_probs[cur][next_node])
        next_state=(next_node,dest,self.die)
        self.curr_rewards += edge_reward
        self.curr_prob *= prob
        self.steps += 1
        self.path_length += G[cur][next_node]['e_l']
        # print('len: ',G[cur][next_node]['e_l'],cur,next_node,edge_reward)
        # exit(0)
        return edge_reward, next_state,G[cur][next_node]['e_l'], done



class PolicyNetwork(tf.keras.Model):
    def __init__(self, num_actions,learning_rate=0.001, seed=None):
        super(PolicyNetwork, self).__init__()
        self.initializer = keras.initializers.GlorotUniform(seed=seed)
        self.dense1 =Dense(512, activation='relu',kernel_initializer=self.initializer)
        self.dense2 =Dense(1024, activation='relu',kernel_initializer=self.initializer)
        self.output_layer =Dense(num_actions, activation='softmax',kernel_initializer=self.initializer)
        self.optimizer = Adam(learning_rate=learning_rate)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)
    
    def update(self, state, action):
        with tf.GradientTape() as tape:
            action_prob = self(state)
            action_mask = tf.one_hot(action, depth=num_actions)
            picked_action_prob = tf.reduce_sum(action_prob * action_mask, axis=1)
            loss = -tf.reduce_sum(tf.math.log(picked_action_prob ))+sum(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
    def reinforce_update(self,state,action,target):
        with tf.GradientTape() as tape:
            action_prob = self(state)
            action_mask = tf.one_hot(action, depth=num_actions)
            picked_action_prob = tf.reduce_sum(action_prob * action_mask, axis=1)
            loss = tf.reduce_sum(-tf.math.log(picked_action_prob + 1e-6) * target) + sum(
                tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
    
    def predict(self, state):
        action_prob = self(state, training=False)
        return action_prob.numpy()
    
def remove_cycles(path):
    seen = set()
    result = []

    for node in path:
        if node not in seen:
            seen.add(node)
            result.append(node)
        else:
            # Break the cycle by removing the repeated nodes
            index = result.index(node)
            result = result[:index + 1]
            break

    return result

def process_path(path, source, destination):
    # Find the last occurrence of the source node
    last_source_index = len(path) - 1 - path[::-1].index(source)

    # Extract the portion from the last occurrence of source to the destination
    processed_path = path[last_source_index:]

    return processed_path



def get_actions_from_shortest_path(graph, shortest_path):
    actions = []
    rewards=[]

    for i in range(len(shortest_path) - 1):
        source_node = shortest_path[i]
        target_node = shortest_path[i + 1]
        # Assuming there's an 'action' attribute associated with each edge

        action = graph[source_node][target_node]['action']
        reward=graph[source_node][target_node]['aqi']+graph[source_node][target_node]['e_g']
        rewards.append(reward)
        actions.append(action)
    return actions,rewards    
def get_available_actions(graph,node):
    ns = list(graph.neighbors(node))
    actions=[]
    for n in  ns:
        action=id_angle(calculate_initial_compass_bearing((nodes[node]['lat'],nodes[node]['lng']),(nodes[n]['lat'],nodes[n]['lng']))) 
        actions.append(action)
    return actions



# Hyperparameters
num_actions = 8  # Number of compass directions
num_episodes = 500



# Initialize the policy network
policy_network = PolicyNetwork(num_actions,seed=42)
with open('node2vec_graph.emd') as file:
    data=file.readlines()
node_embeddings={}
for line in data:
    node_embeddings[(line.split()[0])] = [float(x) for x in line.split()[1:]]
with open('train_data.txt','r') as file:
        data=file.readlines()
# episode=0
# for row in data:
#     episode+=1
#     node=row.split(',')
#     src=node[0]
#     dest=node[1][:-1]
#     dijkstra_path = nx.dijkstra_path(G,src,dest,weight='e_l')
#     dijkstra_path=remove_cycles(dijkstra_path)
#     actions,rewards_data = get_actions_from_shortest_path(G,dijkstra_path)
#     for i in range(len(dijkstra_path)-1):
#         curr=node_embeddings[dijkstra_path[i]]
#         targ=node_embeddings[dest]
#         state=np.expand_dims(np.concatenate((np.asarray(curr), np.asarray(targ) - np.asarray(curr))), axis=0)
#         state= np.squeeze(state)
#         state= np.reshape(state, [-1, 256])    
#         rewards =1
#         action=actions[i]
#         policy_network.update(state,action)

#     if(episode%10==0):
#         print('epi done: ',episode)   
#         print ("reward :",np.mean(rewards_data))
    

# Now the policy network is trained using supervised learning
# You can further fine-tune it using reinforcement learning with rewards

# Continue from the previous code...

# Define the baseline class
class Baseline:
    def __init__(self, decay_rate=0.9):
        self.value = 0.0
        self.decay_rate = decay_rate

    def update(self, rewards):
        self.value = self.decay_rate * self.value + (1 - self.decay_rate) * np.max(rewards)

# Update the training loop
baseline = Baseline()
num_retrain_paths = 10  # T times for every episode
success=0
avg_reward=0
avg_path_length=0
episode=0
num_steps=50
best_path_episode={}
curr_step = 0
rewards_gained = 0.0


# for row in data:
#     episode+=1
#     max_reward = 0
#     max_path = None
#     best_states=None
#     best_actions=None
#     best_env=Env()
#     best=-1
#     # koibar loop chalabo to find dest or done=1
#     for _ in range(num_retrain_paths):
#         env=Env()
#         # state_actions_taken = set()
#         # current_node=src
#         # state_actions_taken_dict = {} 
#         # save_path=[src]
#         total_reward=0
#         path_length = 0.0
#         node=row.split(',')
#         src=node[0]
#         dest=node[1][:-1]
        
#         state_idx=[src,dest]

#         for i in range(num_steps):
#             cur_state=state_embedding(state_idx)
#             action_probs = policy_network.predict(cur_state)

#             # state_key = tuple(state.squeeze().tolist())
#             # # print(state_key)
#             # if state_key not in state_actions_taken_dict:
#             #     state_actions_taken_dict[state_key] = set()

#             neighbor_nodes = list(G.neighbors(state_idx[0]))
#             available_actions=[]
#             for n in  neighbor_nodes:
#                 action=id_angle(calculate_initial_compass_bearing((nodes[state_idx[0]]['lat'],nodes[state_idx[0]]['lng']),(nodes[n]['lat'],nodes[n]['lng']))) 
#                 # don't choose an action that has been chosen before in the same state
#                 if state_idx[0] in env.visited and action not in env.visited[state_idx[0]]:
#                     available_actions.append(action)
#                 elif state_idx[0] not in env.visited:
#                     available_actions.append(action)

#             # available_actions = list(set(get_available_actions(G, current_node)) - state_actions_taken_dict[state_key])            

#             action_probs_avail = []
#             choices_idx = []
#             for c in available_actions:
#                 choices_idx.append(c)
#                 action_probs_avail.append(action_probs[0][c])
#             norm_action_probs = [float(i) / sum(action_probs_avail) for i in action_probs_avail]
#             #ipdb.set_trace()
            
#             # action_probs_avail = np.array([action_probs[0][c] for c in available_actions])
#             # norm_action_probs = action_probs_avail / np.sum(action_probs_avail)

#             # Choose an action based on the probability distribution
#             if len(available_actions) != 0:
#                 if np.random.random() < 0.1:
#                     # With 10% probability, choose a random action
#                     action_chosen = np.random.choice(available_actions)
#                 else:
#                     # Choose an action based on the provided probability distribution
#                     action_chosen = int(np.random.choice(available_actions, p=norm_action_probs))

#                 # state_actions_taken_dict[state_key].add(action_chosen)

#                 edge_reward,new_state,length,done= env.interact(state_idx, action_chosen, 0,norm_action_probs[choices_idx.index(action_chosen)])
#                 total_reward+=edge_reward

#             path_length+=length    

#             if(done==1):
#                 break
#             state_idx=new_state
            
#         if(done==1):
#             print('Successfully reach destination\n')
#             # final_path= process_path(save_path,src,dest)
            
#             # print('final_path: ',final_path)
#             state_batch=[]
#             action_batch = []
#             node_batch = []
#             new_path = []
#             new_path_reward=0.0
#             new_path_avg_reward=0.0
#             curr_node=src
#             # post-process path to remove cycles before rewarding

#             while True:
#                 a = env.last_visited[curr_node][0]
#                 state_batch.append(state_embedding([curr_node,dest]))
#                 node_batch.append(curr_node)
#                 action_batch.append(a)
#                 new_path.append(curr_node)
#                 env.visited[curr_node].discard(a)
#                 transition_reward = env.edge_rewards[curr_node][env.last_visited[curr_node][1]]
#                 new_path_reward+=transition_reward
#                 curr_node = env.last_visited[curr_node][1]
#                 if curr_node == dest:
#                     new_path.append(curr_node)
#                     break
#             # actions,rewards_data = get_actions_from_shortest_path(G,final_path)
#             # for i in range(len(final_path)-1):
#             #     cur_node=final_path[i]
#             #     curr=node_embeddings[final_path[i]]
#             #     next_node=final_path[i+1]
#             #     targ=node_embeddings[dest]
#             #     state=np.expand_dims(np.concatenate((np.asarray(curr), np.asarray(targ) - np.asarray(curr))), axis=0)
#             #     state_batch.append(state)
#             #     total_reward=np.sum(rewards_data)
#             if(new_path_reward>max_reward):
#                 max_reward=new_path_reward
#                 max_path=new_path
#                 best_states=state_batch
#                 best_actions=action_batch
#                 best_env=env.copy()
#                 best=new_path_avg_reward#dont know what is avg
#         else:
#             print('unsuccessful path')

#         # avg_reward+=total_reward
#     # update the best path found?here best is considered best_coun-new_path_crime_count
#     if(max_reward!=0):
#         print('u[date]')   
#         b = avg_reward /(success or 1)
#         discounted_rewards=max_reward
#         # update policy for each state action
#         print('discount_reward: ',max_reward)
#         for s,a in zip(best_states,best_actions):
#             final_reward = discounted_rewards - avg_reward/(success or 1)
#             final_reward = -1 * (max_reward - avg_reward/(success or 1))
#             policy_network.reinforce_update(np.reshape([s], (-1, 256)), 10 * (discounted_rewards), [a])
#         success += 1 
#         avg_reward+=max_reward
#     best_path_episode[episode] = max_reward



# test for validation

def _normalize_probs(p):
    if all(prob == 0 for prob in p):
        return [1.0 / len(p) for prob in p]
    return [float(i) / sum(p) for i in p]

def _create_beam_search_candidate_paths_prob(s, env, choices, scores):
    candidate_paths = []
    norm_action_probs = _normalize_probs(scores)
    chosen_actions = set()
    for i in range(10):
        action_chosen_idx = np.random.choice(len(choices), p=np.asarray(norm_action_probs))
        if action_chosen_idx in chosen_actions:
            continue
        chosen_actions.add(action_chosen_idx)
        new_env = env.copy()
        if len(choices) == 0:
            print('no available action')

        action_chosen = choices[action_chosen_idx]
        reward, new_state, length, done = new_env.interact(s, action_chosen, 0, norm_action_probs[action_chosen_idx])
        candidate_paths.append((new_state, new_env))
        with open('debug.txt','w') as file:
            file.write(f'interect after beam search: {new_env.edge_transition_probs}')

    return candidate_paths



# with open('validation_data.txt','r') as file:
#     data=file.readlines()
# success = 0
# no_success = []
# episode=0
# looped=0
# best_path_lengths={}
# for row in data:
#     episode+=1
#     env=Env()
#     node=row.split(',')
#     src=node[0]
#     dest=node[1][:-1]
#     state_idx=[src,dest]
#     curr_paths = [(state_idx, env)]
#     ended_paths = []
#     for step_count in range(num_steps):
#         next_paths = []
#         for s, e in curr_paths:
#             if e.done:
#                 ended_paths.append((e.curr_prob, s, e))
#                 continue
#             cur_state=state_embedding(s)
#             action_probs = policy_network.predict(cur_state)
#             neighbor_nodes = G.neighbors(s[0])
#             available_actions=[]
#             for n in  neighbor_nodes:
#                 action=id_angle(calculate_initial_compass_bearing((nodes[s[0]]['lat'],nodes[s[0]]['lng']),(nodes[n]['lat'],nodes[n]['lng']))) 
#                 # don't choose an action that has been chosen before in the same state
#                 if s[0] in e.visited and action not in e.visited[s[0]]:
#                     available_actions.append(action)
#                 elif s[0] not in e.visited:
#                     available_actions.append(action)

#             # available_actions = list(set(get_available_actions(G, current_node)) - state_actions_taken_dict[state_key])            

#             action_probs_avail = []
#             choices_idx = []
#             if len(available_actions) == 0:
#                 break
#             for c in available_actions:
#                 choices_idx.append(c)
#                 action_probs_avail.append(action_probs[0][c])
#             norm_action_probs = [float(i) / sum(action_probs_avail) for i in action_probs_avail]
#             candidate_paths = _create_beam_search_candidate_paths_prob(s, e, choices_idx, action_probs_avail)
#             next_paths.extend(candidate_paths)
#         pq = PriorityQueue()
#         for candidate, cand_env in next_paths:
#             pq.put((cand_env.curr_prob, candidate, cand_env))
#             if pq.qsize() > 10 - len(ended_paths):
#                 pq.get()

#         curr_paths = [(candidate, env) for prob, candidate, env in pq.queue]
#         if len(ended_paths) == 10 or len(curr_paths) == 0:
#             break
#     if len(ended_paths) > 0:
#         highest_prob, best_state, best_env = max([(p, s, e) for p, s, e in ended_paths])
#         success += 1
#         print('success')
#         curr_node = src
#         new_path = []
#         new_path_probs = 1.0
#         # new path reward or avg?declare
#         # new patg avgreward path probs 
#         while True:
#             new_path.append(curr_node)
#             new_path_probs *= best_env.edge_transition_probs[curr_node][best_env.last_visited[curr_node][1]]
#             edge_transition_reward = best_env.edge_rewards[curr_node][best_env.last_visited[curr_node][1]]
#             # here adding VG REWARD OR rew
#             curr_node = best_env.last_visited[curr_node][1]
#             if curr_node == dest:
#                 new_path.append(curr_node)
#                 break

#         # again printing calculation
#         # best_path_lengths[episode] = new_path_avg_reward
#         if len(new_path) != len(best_env.path) + 1:
#                 looped += 1

#     else:
#         no_success.append(episode)




# for testing purpose
with open('validation_data.txt','r') as file:
    data=file.readlines()
success = 0
no_success = []
episode=0
avg_len = 0.0
looped = 0            
best_path_lengths = {}
for row in data:
    episode+=1
    env=Env()
    node=row.split(',')
    src=node[0]
    dest=node[1][:-1]
    state_idx=[src,dest]
    ended_paths = []
    curr_paths = [(state_idx, env)]
    for step_count in range(num_steps):
        next_paths = []
        for s, e in curr_paths:
            if e.done:
                ended_paths.append((e.curr_prob, s, e))
                continue
            cur_state=state_embedding(s)
            action_probs = policy_network.predict(cur_state)
            neighbor_nodes = G.neighbors(s[0])
            available_actions=[]
            for n in  neighbor_nodes:
                action=id_angle(calculate_initial_compass_bearing((nodes[s[0]]['lat'],nodes[s[0]]['lng']),(nodes[n]['lat'],nodes[n]['lng']))) 
                # don't choose an action that has been chosen before in the same state
                if s[0] in e.visited and action not in e.visited[s[0]]:
                    available_actions.append(action)
                elif s[0] not in e.visited:
                    available_actions.append(action)

            # available_actions = list(set(get_available_actions(G, current_node)) - state_actions_taken_dict[state_key])            

            action_probs_avail = []
            choices_idx = []
            if len(available_actions) == 0:
                break
            for c in available_actions:
                choices_idx.append(c)
                action_probs_avail.append(action_probs[0][c])
            # norm_action_probs = [float(i) / sum(action_probs_avail) for i in action_probs_avail]
            direction_weights = {d: 1.0 for d in available_actions}

            action_probs_weighted = [p * direction_weights[c] for c, p in zip(available_actions, action_probs_avail)]

            
            candidate_paths = _create_beam_search_candidate_paths_prob(s, e, choices_idx, action_probs_avail)
            next_paths.extend(candidate_paths)
        pq = PriorityQueue()
        for candidate, cand_env in next_paths:
            pq.put((cand_env.curr_prob, candidate, cand_env))
            if pq.qsize() > 5 - len(ended_paths):
                pq.get()

        curr_paths = [(candidate, env) for prob, candidate, env in pq.queue]
        if len(ended_paths) == 5 or len(curr_paths) == 0:
            break
    if len(ended_paths) > 0:
        best_env=None
        max_avg=0.0
        for (p,s,e) in ended_paths:
            avg=2
            if avg > max_avg:
                max_avg = avg
                best_env = e
        success+=1
        curr_node=src
        new_path = []
        new_path_probs = 1.0
        while True:
            new_path.append(curr_node)
            new_path_probs *= best_env.edge_transition_probs[curr_node][best_env.last_visited[curr_node][1]]
            edge_transition_reward = best_env.edge_rewards[curr_node][best_env.last_visited[curr_node][1]]
            # here adding VG REWARD OR rew
            curr_node = best_env.last_visited[curr_node][1]
            if curr_node == dest:
                new_path.append(curr_node)
                break

        # again printing calculation
        # best_path_lengths[episode] = new_path_avg_reward
        if len(new_path) != len(best_env.path) + 1:
                looped += 1

    else:
        no_success.append(episode)