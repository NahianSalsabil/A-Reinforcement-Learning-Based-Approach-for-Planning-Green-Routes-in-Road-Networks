import xmltodict
import pandas as pd
import networkx as nx
import math
import random
from collections import Counter
import matplotlib.pyplot as plt
from node2vec import Node2Vec
import pickle

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
cnt=0
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
            global cnt
            cnt+=1
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

def calc_db_cost_v2(db) -> float:
    """Returns a noise cost for given dB based on a linear scale (dB >= 45 & dB <= 75).
    """
    if db <= 44:
        return 0.0
    db_cost = (db-40) / (75-40)
    return round(db_cost, 3)


def calc_db_cost_v3(db) -> float:
    """Returns a noise cost for given dB: every 10 dB increase doubles the cost (dB >= 45 & dB <= 75).
    """
    if db <= 44:
        return 0.0
    db_cost = pow(10, (0.3 * db)/10)
    return round(db_cost / 100, 3)

def get_noise_range(db: float) -> int:
    """Returns the lower limit of one of the six pre-defined dB ranges based on dB.
    """
    if db >= 70.0:
        return 70
    elif db >= 65.0:
        return 65
    elif db >= 60.0:
        return 60
    elif db >= 55.0:
        return 55
    elif db >= 50.0:
        return 50
    else:
        return 40
cnt=0
for edge in edges:
    
    # Removing curly braces and splitting pairs
    pairs_str = edge['e_n'].strip("{}")
    pairs_list = pairs_str.split(", ")
    if(edge['e_n']=='{}'):
        edge['e_n']=0.0
        cnt+=1
        continue

    # print(pairs_list,len(pairs_list))
    # Converting each pair string into a tuple of integers and floats
    result_pairs = [(get_noise_range(float(pair.split(":")[0].replace(' ',''))), float(pair.split(":")[1])) for pair in pairs_list]

    # Calculating the noise cost for each pair
    noise_costs = [(calc_db_cost_v2(pair[0]),pair[1]) for pair in result_pairs]

    db_distance_cost = sum([db * length for db, length in noise_costs])
    total_length = sum(pair[1] for pair in result_pairs)
    edge['e_n']= round(db_distance_cost / total_length, 3) 

print('cnt: ',cnt)
# print(len(nodes))
# for node in nodes.keys():
#     print(node)
#     break
# print(len(edges))
# print(len(nodes.keys()))
# count=0
# for n in neighbor.keys():
#     if( n=='n3418'):
#         print(n,neighbor[n])

StreetGraph = nx.Graph()
for node in nodes.keys():
  StreetGraph.add_node(node,lat=nodes[node]['lat'],lng=nodes[node]['lng'])
# count = 0
for edge in edges:
    StreetGraph.add_edge(edge['u'], edge['v'] ,e_ii=edge['e_ii'],e_l=float(edge['e_l']),e_b_st=edge['e_b_st'],
                         e_b_ab=edge['e_b_ab'],e_bsf=edge['e_bsf'],e_n=edge['e_n'],
                         e_uv=edge['e_uv'],e_iw=float(edge['e_iw']),e_g=float(edge['e_g']),action=edge['action'])
  

# Custom weight function using the desired edge attribute
def weight_function(u, v, data):
    # Choose the desired attribute for weight, for example, 'e_l'
    attribute_key = 'e_l'
    return float(data.get(attribute_key, 1.0))  # 1.0 as default if attribute is not present

# # Find the shortest path based on the desired edge attribute
# source_node = 'n4989'
# target_node = 'n1001'
# shortest_path = nx.shortest_path(StreetGraph, source_node, target_node, weight=weight_function)


def generate_random_dataset(graph, hop_distance, num_samples):
    dataset = []
    used_sources = set()
    for _ in range(num_samples):
        available_sources = set(graph.nodes()) - used_sources
        if not available_sources:
            break  # If all nodes have been used, exit the loop
        source_node = random.choice(list(available_sources))
        
        # Add the source node to the set of used sources
        used_sources.add(source_node)
        

        # Find neighbors within the specified hop distance
        neighbors = nx.single_source_shortest_path_length(graph, source_node, cutoff=hop_distance)
        

        for target_node, distance in neighbors.items():
            # Skip the source node
            if source_node != target_node and distance == hop_distance:
                dataset.append((source_node,target_node))

    return dataset

dataset=(generate_random_dataset(StreetGraph,15,5000))
with open('train_15.txt','w') as file:
    for row in dataset:
        file.write(row[0]+','+row[1]+'\n')
dataset=(generate_random_dataset(StreetGraph,10,5000))
with open('train_10.txt','w') as file:
    for row in dataset:
        file.write(row[0]+','+row[1]+'\n')
dataset=(generate_random_dataset(StreetGraph,20,5000))
with open('train_20.txt','w') as file:
    for row in dataset:
        file.write(row[0]+','+row[1]+'\n')

# for row in dataset:
#     print(row)
# # print(StreetGraph['n3956']['n5888'])
# # print(StreetGraph['n3956']['n3233'])

# print(nx.dijkstra_path(StreetGraph,'n449','n28',weight='e_iw'))
# print(nx.dijkstra_path(StreetGraph,'n449','n28',weight='e_g'))
# print(neighbor['n449'])

# # G = nx.Graph()
# # G.add_edge('A', 'B', weight=1,dis=9)
# # G.add_edge('B', 'C', weight=2,dis=9)
# # G.add_edge('A', 'C', weight=2,dis=19)
# # print(G['A']['B'])

# # shortest_path = nx.dijkstra_path(G, source='A', target='C', weight='dis')
# # print(shortest_path)

file_path = 'aqi_2020-10-25T14.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

def add_aqi_values_to_edges(graph, aqi_df):
    
    for u, v, data in graph.edges(data=True):
        e_ii_value = data['e_ii']
        # print(aqi_df['id_ig'].values[0])
        if float(e_ii_value) in aqi_df['id_ig'].values:
            # print('yes')
            result_row = aqi_df[aqi_df['id_ig'] == float(e_ii_value)]
            aqi_value= result_row['aqi'].iloc[0]
            graph[u][v]['aqi'] = 1-((aqi_value-1.4)/(1.8-1.4)) #for aqi reversed
    return graph

StreetGraph=add_aqi_values_to_edges(StreetGraph,df)

# with open('features.txt', "w") as file:
#     for u, v, data in StreetGraph.edges(data=True):
#         file.write(str(data['e_ii'])+','+str(data['e_g'])+','+str(data['aqi'])+'\n')


# print(StreetGraph['n278']['n280'])
# cycles = list(nx.algorithms.cycles.simple_cycles(StreetGraph))

# Count the number of cycles
# num_cycles = len(cycles)
# print(num_cycles)

# with open('train.txt','r') as file:
#     data=file.readlines()
# print(len(data))
# cnt=0
# c=0
# for row in data:
#     node=row.split(',')
#     src=node[0]
#     dest=node[1]
#     print(src,dest)
    
#     print('done')
#     if(len(all_paths)>1):
#         print('yeap')
#         cnt+=1
#     c+=1
#     if(c==500):break
# print(cnt)

# all_paths = list(nx.all_simple_paths(StreetGraph, 'n422','n946'))
# print(len(all_paths))



sorted_nodes = sorted(StreetGraph.nodes(),key=lambda x: int(x[1:]))
sorted_edges = sorted(StreetGraph.edges(data=True), key=lambda x: x[0])

# Select the first 50 nodes and the first 200 edges
selected_nodes = sorted_nodes[:1000]
selected_edges = sorted_edges[:50]

# Create a new graph with the selected nodes and edges
n=2000
subgraph_nodes = sorted_nodes[:n]
subgraph_edges = [(u, v) for u, v in StreetGraph.edges() if u in subgraph_nodes and v in subgraph_nodes]

subgraph = nx.Graph()
subgraph.add_nodes_from(subgraph_nodes)
subgraph.add_edges_from(subgraph_edges)
print(nx.dijkstra_path(StreetGraph,'n278','n626',weight='e_l'))

# # Draw the subgraph
# pos = nx.spring_layout(subgraph)
# nx.draw(subgraph, pos, with_labels=True, node_size=n, font_size=8)
# plt.show()

# dataset=generate_random_dataset(subgraph,5,700)
# print(len(dataset))
# with open('train.txt','w') as file:
#     for row in dataset:
#         file.write(row[0]+','+row[1]+'\n')

# with open('train.txt', 'r') as input_file:
#     lines = input_file.readlines()

# # Randomly sample 500 lines
# sampled_lines = random.sample(lines, 5000)

# # Remove sampled lines from the original list
# lines = [line for line in lines if line not in sampled_lines]

# # Write the sampled lines to the first output file
# with open('train_data.txt', 'w') as output_file_1:
#     output_file_1.writelines(sampled_lines)

# # Randomly sample 200 lines from the remaining lines
# sampled_lines_2 = random.sample(lines, 1500)

# # Write the second set of sampled lines to the second output file
# with open('validation_data.txt', 'w') as output_file_2:
#     output_file_2.writelines(sampled_lines_2)
# lines2 = [line for line in lines if line not in sampled_lines and line not in sampled_lines_2]

# sampled_lines_3 = random.sample(lines2, 750)
# with open('test_data.txt', 'w') as output_file_3:
#     output_file_3.writelines(sampled_lines_3)

# node2vec = Node2Vec(StreetGraph, dimensions=128, walk_length=30, num_walks=200, workers=4)

# # Generate walks and learn embeddings
# model = node2vec.fit(window=10, min_count=1, batch_words=4)

# # Access the embeddings
# embeddings = {str(node): model.wv[str(node)] for node in StreetGraph.nodes()}
# # print(embeddings)
# print(len(embeddings))
# keys = embeddings.keys()
# for key in keys:
#   first_key = key
#   break
# print(first_key)
# emd_file_path = "node2vec_graph.emd"

# # Open the .emd file in write mode
# with open(emd_file_path, "w") as emd_file:
#     # Write the number of nodes and dimensionality of embeddings as a header

#     num_nodes, embedding_dim = len(embeddings), len(embeddings[first_key])
#     emd_file.write(f"{num_nodes} {embedding_dim}\n")

#     # Write node embeddings
#     for node, embedding in embeddings.items():
#         embedding_str = " ".join(map(str, embedding))
#         emd_file.write(f"{node} {embedding_str}\n")

# print(f"Node embeddings saved to {emd_file_path}")

# with open('full_graph.gpickle', "wb") as file:
#     pickle.dump(StreetGraph, file)

# with open('sub_graph.gpickle', "wb") as file:
#     pickle.dump(subgraph, file)

# neighbors = nx.shortest_path(StreetGraph, 'n626','n278', weight='e_l')
# src='n626'
# print(StreetGraph['n626']['n1447'])

# with open('features.txt', "w") as file:
#     for u, v, data in StreetGraph.edges(data=True):
#         file.write(str(data['e_ii'])+','+str(data['e_g'])+','+str(data['aqi'])+','+str(data['e_n'])+'\n')

# # # Read data from the file
# file_path = 'features.txt'
# with open(file_path, 'r') as file:
#     lines = file.readlines()

# # Process the data
# data = []
# noise_values = []

# for line in lines:
#     parts = line.strip().split(',')
#     id_value = int(parts[0])
#     gvi_value = float(parts[1])
#     aqi_value = float(parts[2])
#     noise_value = float(parts[3][:-1]) if len(parts) == 4 else 0.0  # Handle missing noise values
#     data.append((id_value, gvi_value, aqi_value, noise_value))
#     if noise_value != 0.0:
#         noise_values.append(noise_value)

# # Calculate the mean of noise values
# mean_noise = sum(noise_values) / len(noise_values) if noise_values else 0.0

# # Replace zero noise values with the mean
# for i in range(len(data)):
#     if data[i][3] == 0.0:
#         data[i] = (data[i][0], data[i][1], data[i][2],1- mean_noise)
#     else:
#         data[i] = (data[i][0], data[i][1], data[i][2],1- data[i][3])

# # Write the updated data back to the file
# with open(file_path, 'w') as file:
#     for entry in data:
#         file.write(','.join(map(str, entry)) + '\n')

# print("Noise values replaced with mean:", mean_noise)

# for u, v, data in StreetGraph.edges(data=True):
#     if data['e_n'] == 0.0:
#         data['e_n'] = 1-mean_noise
#     else:
#         data['e_n'] = 1-data['e_n']

# with open('full_graph.gpickle', "wb") as file:
#     pickle.dump(StreetGraph, file)

# # print(StreetGraph['n2377']['n2376'])n5902,n4988
# print('shortest path :',nx.dijkstra_path(StreetGraph,'n5902','n4988',weight='e_l'))
# # print('shortest path :',(StreetGraph.nodes['n1263']))
# # noise:
# # min-0.529
# # max-.857
# # avg=.444

# def calculate_percentage(data, column_index, lower_bound, upper_bound):
#     total_count = len(data)
#     selected_values = [float(row[column_index].strip()) for row in data if lower_bound <= float(row[column_index].strip()) <= upper_bound]
#     selected_count = len(selected_values)
#     percentage = (selected_count / total_count) * 100
#     return percentage


# # Read the features.txt file
# with open('features.txt', 'r') as file:
#     lines = file.readlines()

# # Parse the data
# data = [line.strip().split(',') for line in lines]

# range_divisions = [0, 0.25, 0.5, 0.75, 1]

# # Calculate and print the percentage for each column and range
# for column_index in range(1, 4):  # Starting from the second column
#     print(f"Column {column_index}:")
#     for i in range(len(range_divisions) - 1):
#         lower, upper = range_divisions[i], range_divisions[i + 1]
#         percentage = calculate_percentage(data, column_index, lower, upper)
#         print(f"  {lower}-{upper}: {percentage:.2f}%")


# # Read data from the file
# with open('features.txt', 'r') as file:
#     lines = file.readlines()

# # Extract relevant columns and convert to float
# green_view_index = [float(line.split(',')[1]) for line in lines]
# aqi = [float(line.split(',')[2]) for line in lines]
# noise=[float(line.split(',')[3][:-1]) for line in lines]

# # Calculate min, max, and average
# min_gvi = min(green_view_index)
# max_gvi = max(green_view_index)
# avg_gvi = sum(green_view_index) / len(green_view_index)

# min_aqi = min(aqi)
# max_aqi = max(aqi)
# avg_aqi = sum(aqi) / len(aqi)

# min_noise = min(noise)
# max_noise = max(noise)  
# avg_noise = sum(noise) / len(noise) 



# # Print the results
# print("Green View Index:")
# print("Min:", min_gvi)
# print("Max:", max_gvi)
# print("Average:", avg_gvi)

# print("\nAir Quality Index:")
# print("Min:", min_aqi)
# print("Max:", max_aqi)
# print("Average:", avg_aqi)

# print("\nNoise:")       

# print("Min:", min_noise)
# print("Max:", max_noise)
# print("Average:", avg_noise)

# def check_cycles(path):
#     rel_ents=path
#     entity_stats = Counter(path).items()
#     duplicate_ents = [item for item in entity_stats if item[1] != 1]
#     duplicate_ents.sort(key=lambda t: t[1], reverse=True)
#     if len(duplicate_ents)==0:
#       return False
#     return True

# with open('test_data_1000_5.txt','r') as file:
#         data=file.readlines()
# cnt=0
# cnt_d=0
# for row in data:
#     node=row.split(',')
#     src=node[0]
#     dest=node[1][:-1]
#     d=nx.dijkstra_path(StreetGraph,src,dest,weight='e_l')
#     if(len(d)>6):
#         cnt_d+=1
#     paths = list(nx.all_simple_paths(StreetGraph, source=src, target=dest, cutoff=5))
#     if(len(paths)==1):
#         cnt+=1
# print(len(data))
# print(cnt)
# print(cnt_d)


# with open('test_data_1000_10.txt','r') as file:
#         data=file.readlines()
# cnt=0
# cnt_d=0
# for row in data:
#     node=row.split(',')
#     src=node[0]
#     dest=node[1][:-1]
#     d=nx.dijkstra_path(StreetGraph,src,dest,weight='e_l')
#     if(len(d)>11):
#         cnt_d+=1
#     paths = list(nx.all_simple_paths(StreetGraph, source=src, target=dest, cutoff=10))
#     if(len(paths)==1):
#         cnt+=1
# print(len(data))
# print(cnt)
# print(cnt_d)

# with open('test_data_2000_5.txt','r') as file:
#         data=file.readlines()
# cnt=0
# cnt_d=0
# for row in data:
#     node=row.split(',')
#     src=node[0]
#     dest=node[1][:-1]
#     d=nx.dijkstra_path(StreetGraph,src,dest,weight='e_l')
#     if(len(d)>6):
#         cnt_d+=1
#     paths = list(nx.all_simple_paths(StreetGraph, source=src, target=dest, cutoff=5))
#     if(len(paths)==1):
#         cnt+=1
# print(len(data))
# print(cnt)
# print(cnt_d)

# with open('test_data_2000_10.txt','r') as file:
#         data=file.readlines()
# cnt=0
# cnt_d=0
# for row in data:
#     node=row.split(',')
#     src=node[0]
#     dest=node[1][:-1]
#     d=nx.dijkstra_path(StreetGraph,src,dest,weight='e_l')
#     if(len(d)>11):
#         cnt_d+=1
#     paths = list(nx.all_simple_paths(StreetGraph, source=src, target=dest, cutoff=10))
#     if(len(paths)==1):
#         cnt+=1
# print(len(data))
# print(cnt)
# print(cnt_d)


