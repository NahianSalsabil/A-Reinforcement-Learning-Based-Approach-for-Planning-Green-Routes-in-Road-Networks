import xmltodict
import pandas as pd
import networkx as nx
import math
import random
from collections import Counter
import matplotlib.pyplot as plt
from node2vec import Node2Vec
import pickle


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
        src=int(src[1:])
        dest=int(dest[1:])
        print(type(src),type(dest))

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


# def calc_db_cost_v3(db) -> float:
#     """Returns a noise cost for given dB: every 10 dB increase doubles the cost (dB >= 45 & dB <= 75).
#     """
#     if db <= 44:
#         return 0.0
#     db_cost = pow(10, (0.3 * db)/10)
#     return round(db_cost / 100, 3)

# def get_noise_range(db: float) -> int:
#     """Returns the lower limit of one of the six pre-defined dB ranges based on dB.
#     """
#     if db >= 70.0:
#         return 70
#     elif db >= 65.0:
#         return 65
#     elif db >= 60.0:
#         return 60
#     elif db >= 55.0:
#         return 55
#     elif db >= 50.0:
#         return 50
#     else:
#         return 40
# cnt=0
# for edge in edges:
    
#     # Removing curly braces and splitting pairs
#     pairs_str = edge['e_n'].strip("{}")
#     pairs_list = pairs_str.split(", ")
#     if(edge['e_n']=='{}'):
#         edge['e_n']=0.0
#         cnt+=1
#         continue

#     # print(pairs_list,len(pairs_list))
#     # Converting each pair string into a tuple of integers and floats
#     result_pairs = [(get_noise_range(float(pair.split(":")[0].replace(' ',''))), float(pair.split(":")[1])) for pair in pairs_list]

#     # Calculating the noise cost for each pair
#     noise_costs = [(calc_db_cost_v3(pair[0]),pair[1]) for pair in result_pairs]

#     db_distance_cost = sum([db * length for db, length in noise_costs])
#     total_length = sum(pair[1] for pair in result_pairs)
#     edge['e_n']= round(db_distance_cost / total_length, 3) 

# print('cnt: ',cnt)


# StreetGraph = nx.Graph()
# for node in nodes.keys():
#   StreetGraph.add_node(node,latitude=nodes[node]['lat'],longitude=nodes[node]['lng'])
# # count = 0
# for edge in edges:
#     StreetGraph.add_edge(edge['u'], edge['v'] ,e_ii=edge['e_ii'],e_l=float(edge['e_l']),e_b_st=edge['e_b_st'],
#                          e_b_ab=edge['e_b_ab'],e_bsf=edge['e_bsf'],e_n=edge['e_n'],
#                          e_uv=edge['e_uv'],e_iw=float(edge['e_iw']),e_g=float(edge['e_g']))
  

# file_path = 'aqi_2020-10-25T14.csv'

# # Read the CSV file into a DataFrame
# df = pd.read_csv(file_path)

# def add_aqi_values_to_edges(graph, aqi_df):
    
#     for u, v, data in graph.edges(data=True):
#         e_ii_value = data['e_ii']
#         # print(aqi_df['id_ig'].values[0])
#         if float(e_ii_value) in aqi_df['id_ig'].values:
#             # print('yes')
#             result_row = aqi_df[aqi_df['id_ig'] == float(e_ii_value)]
#             aqi_value= result_row['aqi'].iloc[0]
#             graph[u][v]['aqi'] = aqi_value
#     return graph

# StreetGraph=add_aqi_values_to_edges(StreetGraph,df)

# import numpy as np

# # Calculate the mean of 'e_n' excluding NaN values
# e_n_values = [data['e_n'] for _, _, data in StreetGraph.edges(data=True) if not np.isnan(data['e_n'])]
# mean_e_n = np.mean(e_n_values)

# # Iterate over all edges and fill NaN values with the mean
# for u, v, data in StreetGraph.edges(data=True):
#     if np.isnan(data['e_n']) or data['e_n'] == 0.0:
#         StreetGraph[u][v]['e_n'] = mean_e_n

# import numpy as np

# def normalize_attribute(graph, attribute):
#     # Extract all non-NaN values of the attribute from the graph
#     values = [data[attribute] for _, _, data in graph.edges(data=True) if not np.isnan(data[attribute])]
    
#     # Calculate the minimum and maximum values
#     min_value = np.min(values)
#     max_value = np.max(values)
    
#     # Avoid division by zero in case all values are the same
#     if min_value == max_value:
#         return graph
    
#     # Normalize the attribute in all edges
#     for u, v, data in graph.edges(data=True):
#         if not np.isnan(data[attribute]):
#             # Min-max normalization scaled to 0-10
#             data[attribute] = (((data[attribute] - min_value) / (max_value - min_value)) * 9) + 1
    
#     return graph

# # Normalize 'e_n'
# StreetGraph = normalize_attribute(StreetGraph, 'e_n')

# # Normalize 'e_g' if it exists in your edge data
# StreetGraph = normalize_attribute(StreetGraph, 'e_g')

# # Normalize 'aqi' if it exists in your edge data
# StreetGraph = normalize_attribute(StreetGraph, 'aqi')





# with open('normalized_graph.gpickle', "wb") as file:
#     pickle.dump(StreetGraph, file)


# import networkx as nx
# import numpy as np

# def normalize_and_reverse(graph, attributes):
#     # Create a deep copy of the graph to not modify the original
#     new_graph = nx.Graph(graph)
    
#     # Normalize attributes to 1-10 and reverse 'aqi' and 'e_n'
#     for attr in attributes:
        
#         for u, v, data in new_graph.edges(data=True):
#             if attr in data:

#                 # Reverse 'aqi' and 'e_n' by subtracting from 11
#                 if attr in ['aqi', 'e_n']:
#                     data[attr] = 11 - data[attr]

#     return new_graph

# # Attributes to normalize
# attributes = ['aqi', 'e_n', 'e_g']

# # Normalize and reverse 'aqi' and 'e_n' in the new graph
# NewGraph = normalize_and_reverse(StreetGraph, attributes)
# with open('reversed_normalized_graph.gpickle', "wb") as file:
#     pickle.dump(NewGraph, file)

# # Optional: Verify the updated values in NewGraph
# for _, _, data in NewGraph.edges(data=True):
#     print(f"AQI: {data.get('aqi', 'N/A')}, e_n: {data.get('e_n', 'N/A')}, e_g: {data.get('e_g', 'N/A')}")

# with open('features.txt', "w") as file:
#     for u, v, data in StreetGraph.edges(data=True):
#         file.write(str(data['e_ii'])+','+str(data['e_g'])+','+str(data['aqi'])+','+str(data['e_n'])+'\n')

StreetGraph = pd.read_pickle("reversed_normalized_graph.gpickle")
print(nx.shortest_path(StreetGraph, source=1322, target=5219, weight='e_l'))

# # print('print nodes in ascending order: ',sorted(G.nodes()))
# node2vec = Node2Vec(StreetGraph, dimensions=128, walk_length=30, num_walks=200, workers=4)   
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
# emd_file_path = "node2vec_graph_num.emd"

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

# print('print nodes in sorted order: ',sorted(StreetGraph.nodes()))
# print(len(StreetGraph.edges()))