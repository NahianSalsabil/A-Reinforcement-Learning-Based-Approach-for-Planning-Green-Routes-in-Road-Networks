import ast
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
import random
import math
import networkx as nx


inp='exp1_train2.csv'
inp2='exp2_train.csv'
inp3='exp3_train.csv'
inp4='exp4_train.csv'
inp5='exp5_train.csv'
inp6='exp6_train.csv'
inp7='exp7_train.csv'
inp8='exp8_train.csv'
inp9='exp9_train.csv'
inp10='exp10_train.csv'
inp11='exp11_train.csv'
inp12='exp12_train.csv'
inp13='exp13_train.csv'
inp14='exp14_train.csv'
dd='r2/train/'
dt='r2/train/'
dtt='r2/train'
rew1=0
w=.001
def calculate_combined_reward(aqi, green, noise,plen,issuccess,norm):
    a=b=c=0
    if aqi >=7:
        a = 100
    elif aqi >=4:
        a = 60
    elif aqi >=0:
        a = 30
    if green >=7:
        b = 100
    elif green >=4:
        b = 60
    elif green >=0:
        b = 30
    if noise >=7:
        c = 100
    elif noise >=4:
        c = 60
    elif noise >=0:
        c = 30
    x=math.log(a)*aqi
    y=math.log(b)*green
    z=math.log(c)*noise
    fl=((y+x+z)*20)+(plen/norm)*4+issuccess
    return fl



def calculate_combined_reward2(aqi, green, noise,p,p1):
    global w
    a=b=c=0
    if aqi >=7:
        a = 100
    elif aqi >=4:
        a = 60
    elif aqi >=0:
        a = 30
    if green >=7:
        b = 100
    elif green >=4:
        b = 60
    elif green >=0:
        b = 30
    if noise >=7:
        c = 100
    elif noise >=4:
        c = 60
    elif noise >=0:
        c = 30
    x=math.log(a)*aqi
    y=math.log(b)*green
    z=math.log(c)*noise

    fl=((y+x+z)*20)+w*(p/p1)
    # print('fl: ',fl)
    return fl

def lat_long_to_cartesian(lat, lon, R=6371):  # R is the Earth's radius in kilometers
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    x = R * math.cos(lat_rad) * math.cos(lon_rad)
    y = R * math.cos(lat_rad) * math.sin(lon_rad)
    return x, y

def euclidean_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def find_straight_line_distance(node1,node2,dest):
    # print('node1,node2,dest: ',node1,node2,dest)
    cart1 = lat_long_to_cartesian(G.nodes[node1]['latitude'], G.nodes[node1]['longitude'])
    cart2 = lat_long_to_cartesian(G.nodes[dest]['latitude'], G.nodes[dest]['longitude'])
    cart3 = lat_long_to_cartesian(G.nodes[node2]['latitude'], G.nodes[node2]['longitude'])
    distance1= euclidean_distance(cart1, cart2)
    distance2= euclidean_distance(cart3, cart2)
    # print('distances: ',distance2-distance1)
    # print('distance: ',distance1,distance2,distance2-distance1)
    return distance2-distance1

def calculatePathValues(pathArr):
    # print("Path: ",pathArr)
    aqi_random=[]
    green_random=[]
    noise_random=[]
    total_reward=[]
    total_len=0
    sp=nx.shortest_path_length(G,source=pathArr[0],target=pathArr[-1],weight='e_l')
    global rew1,w

    # calculate total length from path[0] to path[-1]
    for i in range(len(pathArr)-1):
        total_len+= G[pathArr[i]][pathArr[i+1]]['e_l']

    for i in range(len(pathArr)-1):
        aqi_random.append( G[pathArr[i]][pathArr[i+1]]['aqi'])
        green_random.append( G[pathArr[i]][pathArr[i+1]]['e_g'])
        noise_random.append( G[pathArr[i]][pathArr[i+1]]['e_n'])
        if rew1==3 or rew1==5 or rew1==5 or rew1==7 or rew1==8 or rew1==9  or rew1==11 or rew1==12 or rew1==13 or rew1==14:
            if rew1==3 or rew1==5 or rew1==9:
                w=.001
            elif rew1==7 :
                w=.1
            elif rew1==8:
                w=.000001
            elif rew1==11:
                w=.000000001
            else:
                w=.000001
            total_reward.append(calculate_combined_reward2(G[pathArr[i]][pathArr[i+1]]['aqi'],G[pathArr[i]][pathArr[i+1]]['e_g'],G[pathArr[i]][pathArr[i+1]]['e_n'],sp,total_len))
        else:
            pl=find_straight_line_distance(pathArr[i],pathArr[i+1],pathArr[-1])
            if pathArr[i+1]==pathArr[-1]:
                iss=30
            else :
                iss=0
            total_reward.append(calculate_combined_reward(G[pathArr[i]][pathArr[i+1]]['aqi'],G[pathArr[i]][pathArr[i+1]]['e_g'],G[pathArr[i]][pathArr[i+1]]['e_n'],pl,iss,sp))
            
    return aqi_random,green_random,noise_random,total_reward



def read_paths_from_file(file_path):
    with open(file_path, 'r') as file:
        # Read the entire content of the file as a single string
        content = file.readlines()

    return content

# Assuming your file is named 'paths.txt' and located in the current directory



G = pd.read_pickle("normalized_graph.gpickle")
# print len of total edge and nodes
print(len(G.edges),len(G.nodes))
exit(0)
reverse_G = pd.read_pickle("reversed_normalized_graph.gpickle")  


paths = read_paths_from_file(inp)
aqi_1_mean=[]
green_1_mean=[]
noise_1_mean=[]
aqi_1_min=[]
green_1_min=[]
noise_1_min=[]
aqi_1_max=[]
green_1_max=[]
noise_1_max=[]
aqi_2_mean=[]
green_2_mean=[]
noise_2_mean=[]
aqi_2_min=[]
green_2_min=[]
noise_2_min=[]
aqi_2_max=[]
green_2_max=[]
noise_2_max=[]
aqi_3_mean=[]
green_3_mean=[]
noise_3_mean=[]
aqi_3_min=[]
green_3_min=[]
noise_3_min=[]
aqi_3_max=[]
green_3_max=[]
noise_3_max=[]
aqi_4_mean=[]
green_4_mean=[]
noise_4_mean=[]
aqi_4_min=[]
green_4_min=[]
noise_4_min=[]
aqi_4_max=[]
green_4_max=[]
noise_4_max=[]
aqi_5_mean=[]
green_5_mean=[]
noise_5_mean=[]
aqi_5_min=[]
green_5_min=[]
noise_5_min=[]
aqi_5_max=[]
green_5_max=[]
noise_5_max=[]
aqi_6_mean=[]
green_6_mean=[]
noise_6_mean=[]
aqi_6_min=[]
green_6_min=[]
noise_6_min=[]
aqi_6_max=[]
green_6_max=[]
noise_6_max=[]
aqi_7_mean=[]
green_7_mean=[]
noise_7_mean=[]
aqi_7_min=[]
green_7_min=[]
noise_7_min=[]
aqi_7_max=[]
green_7_max=[]
noise_7_max=[]
aqi_8_mean=[]
green_8_mean=[]
noise_8_mean=[]
aqi_8_min=[]
green_8_min=[]
noise_8_min=[]
aqi_8_max=[]
green_8_max=[]
noise_8_max=[]
aqi_9_mean=[]
green_9_mean=[]
noise_9_mean=[]
aqi_9_min=[]
green_9_min=[]
noise_9_min=[]
aqi_9_max=[]
green_9_max=[]
noise_9_max=[]
aqi_10_mean=[]
green_10_mean=[]
noise_10_mean=[]
aqi_10_min=[]
green_10_min=[]
noise_10_min=[]
aqi_10_max=[]
green_10_max=[]
noise_10_max=[]
aqi_11_mean=[]
green_11_mean=[]
noise_11_mean=[]
aqi_11_min=[]
green_11_min=[]
noise_11_min=[]
aqi_11_max=[]
green_11_max=[]
noise_11_max=[]
aqi_12_mean=[]
green_12_mean=[]
noise_12_mean=[]
aqi_12_min=[]
green_12_min=[]
noise_12_min=[]
aqi_12_max=[]
green_12_max=[]
noise_12_max=[]
aqi_13_mean=[]
green_13_mean=[]
noise_13_mean=[]
aqi_13_min=[]
green_13_min=[]
noise_13_min=[]
aqi_13_max=[]
green_13_max=[]
noise_13_max=[]
aqi_14_mean=[]
green_14_mean=[]
noise_14_mean=[]
aqi_14_min=[]
green_14_min=[]
noise_14_min=[]
aqi_14_max=[]
green_14_max=[]
noise_14_max=[]

r_1_mean=[]
r_2_mean=[]
r_3_mean=[]
r_4_mean=[]
r_5_mean=[]
r_6_mean=[]
r_7_mean=[]
r_8_mean=[]
r_9_mean=[]
r_10_mean=[]
r_11_mean=[]
r_12_mean=[]
r_13_mean=[]
r_14_mean=[]



for path in paths:
    p=""
    for i in range(len(path)):
        p+=path[i]
    p=p.split(',')
    for i in range(len(p)):
        p[i]=int(p[i])
    src=p[0]
    dest=p[-1]
    rew1=1
    aqi_1,green_1,noise_1,r_1=calculatePathValues(p)
    aqi_1_mean.append(np.mean(aqi_1))
    green_1_mean.append(np.mean(green_1))
    noise_1_mean.append(np.mean(noise_1))
    aqi_1_min.append(np.min(aqi_1))
    green_1_min.append(np.min(green_1))
    noise_1_min.append(np.min(noise_1))
    aqi_1_max.append(np.max(aqi_1))
    green_1_max.append(np.max(green_1))
    noise_1_max.append(np.max(noise_1))
    r_1_mean.append(np.mean(r_1))

paths = read_paths_from_file(inp2)
for path in paths:
    p=""
    for i in range(len(path)):
        p+=path[i]
    p=p.split(',')
    for i in range(len(p)):
        p[i]=int(p[i])
    src=p[0]
    dest=p[-1]
    rew1=2
    aqi_2,green_2,noise_2,r_2=calculatePathValues(p)
    aqi_2_mean.append(np.mean(aqi_2))
    green_2_mean.append(np.mean(green_2))
    noise_2_mean.append(np.mean(noise_2))
    aqi_2_min.append(np.min(aqi_2))
    green_2_min.append(np.min(green_2))
    noise_2_min.append(np.min(noise_2))
    aqi_2_max.append(np.max(aqi_2))
    green_2_max.append(np.max(green_2))
    noise_2_max.append(np.max(noise_2))
    r_2_mean.append(np.mean(r_2))

paths = read_paths_from_file(inp3)
for path in paths:
    p=""
    for i in range(len(path)):
        p+=path[i]
    p=p.split(',')
    for i in range(len(p)):
        p[i]=int(p[i])
    src=p[0]
    dest=p[-1]
    rew1=3
    aqi_3,green_3,noise_3,r_3=calculatePathValues(p)
    r_3_mean.append(np.mean(r_3))
    aqi_3_mean.append(np.mean(aqi_3))
    green_3_mean.append(np.mean(green_3))
    noise_3_mean.append(np.mean(noise_3))
    aqi_3_min.append(np.min(aqi_3))
    green_3_min.append(np.min(green_3))
    noise_3_min.append(np.min(noise_3))
    aqi_3_max.append(np.max(aqi_3))
    green_3_max.append(np.max(green_3))
    noise_3_max.append(np.max(noise_3))


paths = read_paths_from_file(inp4)
for path in paths:
    p=""
    for i in range(len(path)):
        p+=path[i]
    p=p.split(',')
    for i in range(len(p)):
        p[i]=int(p[i])
    src=p[0]
    dest=p[-1]
    rew1=4
    aqi_4,green_4,noise_4,r_4=calculatePathValues(p)
    r_4_mean.append(np.mean(r_4))
    aqi_4_mean.append(np.mean(aqi_4))
    green_4_mean.append(np.mean(green_4))
    noise_4_mean.append(np.mean(noise_4))
    aqi_4_min.append(np.min(aqi_4))
    green_4_min.append(np.min(green_4))
    noise_4_min.append(np.min(noise_4))
    aqi_4_max.append(np.max(aqi_4))
    green_4_max.append(np.max(green_4))
    noise_4_max.append(np.max(noise_4))

paths = read_paths_from_file(inp5)
for path in paths:
    p=""
    for i in range(len(path)):
        p+=path[i]
    p=p.split(',')
    for i in range(len(p)):
        p[i]=int(p[i])
    src=p[0]
    dest=p[-1]
    rew1=5
    aqi_5,green_5,noise_5,r_5=calculatePathValues(p)
    r_5_mean.append(np.mean(r_5))
    aqi_5_mean.append(np.mean(aqi_5))
    green_5_mean.append(np.mean(green_5))
    noise_5_mean.append(np.mean(noise_5))
    aqi_5_min.append(np.min(aqi_5))
    green_5_min.append(np.min(green_5))
    noise_5_min.append(np.min(noise_5))
    aqi_5_max.append(np.max(aqi_5))
    green_5_max.append(np.max(green_5))
    noise_5_max.append(np.max(noise_5))

paths = read_paths_from_file(inp6)
for path in paths:
    p=""
    for i in range(len(path)):
        p+=path[i]
    p=p.split(',')
    for i in range(len(p)):
        p[i]=int(p[i])
    src=p[0]
    dest=p[-1]
    rew1=6
    aqi_6,green_6,noise_6,r_6=calculatePathValues(p)
    r_6_mean.append(np.mean(r_6))
    aqi_6_mean.append(np.mean(aqi_6))
    green_6_mean.append(np.mean(green_6))
    noise_6_mean.append(np.mean(noise_6))
    aqi_6_min.append(np.min(aqi_6))
    green_6_min.append(np.min(green_6))
    noise_6_min.append(np.min(noise_6))
    aqi_6_max.append(np.max(aqi_6))
    green_6_max.append(np.max(green_6))
    noise_6_max.append(np.max(noise_6))

paths = read_paths_from_file(inp7)
for path in paths:
    p=""
    for i in range(len(path)):
        p+=path[i]
    p=p.split(',')
    for i in range(len(p)):
        p[i]=int(p[i])
    src=p[0]
    dest=p[-1]
    rew1=7
    aqi_7,green_7,noise_7,r_7=calculatePathValues(p)
    r_7_mean.append(np.mean(r_7))
    aqi_7_mean.append(np.mean(aqi_7))
    green_7_mean.append(np.mean(green_7))
    noise_7_mean.append(np.mean(noise_7))
    aqi_7_min.append(np.min(aqi_7))
    green_7_min.append(np.min(green_7))
    noise_7_min.append(np.min(noise_7))
    aqi_7_max.append(np.max(aqi_7))
    green_7_max.append(np.max(green_7))
    noise_7_max.append(np.max(noise_7))

paths = read_paths_from_file(inp8)
for path in paths:
    p=""
    for i in range(len(path)):
        p+=path[i]
    p=p.split(',')
    for i in range(len(p)):
        p[i]=int(p[i])
    src=p[0]
    dest=p[-1]
    rew1=8
    aqi_8,green_8,noise_8,r_8=calculatePathValues(p)
    r_8_mean.append(np.mean(r_8))
    aqi_8_mean.append(np.mean(aqi_8))
    green_8_mean.append(np.mean(green_8))
    noise_8_mean.append(np.mean(noise_8))
    aqi_8_min.append(np.min(aqi_8))
    green_8_min.append(np.min(green_8))
    noise_8_min.append(np.min(noise_8))
    aqi_8_max.append(np.max(aqi_8))
    green_8_max.append(np.max(green_8))
    noise_8_max.append(np.max(noise_8))

paths = read_paths_from_file(inp9)
for path in paths:
    p=""
    for i in range(len(path)):
        p+=path[i]
    p=p.split(',')
    for i in range(len(p)):
        p[i]=int(p[i])
    src=p[0]
    dest=p[-1]
    rew1=9
    aqi_9,green_9,noise_9,r_9=calculatePathValues(p)
    r_9_mean.append(np.mean(r_9))
    aqi_9_mean.append(np.mean(aqi_9))
    green_9_mean.append(np.mean(green_9))
    noise_9_mean.append(np.mean(noise_9))
    aqi_9_min.append(np.min(aqi_9))
    green_9_min.append(np.min(green_9))
    noise_9_min.append(np.min(noise_9))
    aqi_9_max.append(np.max(aqi_9))
    green_9_max.append(np.max(green_9))
    noise_9_max.append(np.max(noise_9))

paths = read_paths_from_file(inp10)
for path in paths:
    p=""
    for i in range(len(path)):
        p+=path[i]
    p=p.split(',')
    for i in range(len(p)):
        p[i]=int(p[i])
    src=p[0]
    dest=p[-1]
    rew1=10
    aqi_10,green_10,noise_10,r_10=calculatePathValues(p)
    r_10_mean.append(np.mean(r_10))
    aqi_10_mean.append(np.mean(aqi_10))
    green_10_mean.append(np.mean(green_10))
    noise_10_mean.append(np.mean(noise_10))
    aqi_10_min.append(np.min(aqi_10))
    green_10_min.append(np.min(green_10))
    noise_10_min.append(np.min(noise_10))
    aqi_10_max.append(np.max(aqi_10))
    green_10_max.append(np.max(green_10))
    noise_10_max.append(np.max(noise_10))

paths = read_paths_from_file(inp11)
for path in paths:
    p=""
    for i in range(len(path)):
        p+=path[i]
    p=p.split(',')
    for i in range(len(p)):
        p[i]=int(p[i])
    src=p[0]
    dest=p[-1]
    rew1=11
    aqi_11,green_11,noise_11,r_11=calculatePathValues(p)
    r_11_mean.append(np.mean(r_11))
    aqi_11_mean.append(np.mean(aqi_11))
    green_11_mean.append(np.mean(green_11))
    noise_11_mean.append(np.mean(noise_11))
    aqi_11_min.append(np.min(aqi_11))
    green_11_min.append(np.min(green_11))
    noise_11_min.append(np.min(noise_11))
    aqi_11_max.append(np.max(aqi_11))
    green_11_max.append(np.max(green_11))
    noise_11_max.append(np.max(noise_11))

paths = read_paths_from_file(inp12)
for path in paths:
    p=""
    for i in range(len(path)):
        p+=path[i]
    p=p.split(',')
    for i in range(len(p)):
        p[i]=int(p[i])
    src=p[0]
    dest=p[-1]
    rew1=12
    aqi_12,green_12,noise_12,r_12=calculatePathValues(p)
    r_12_mean.append(np.mean(r_12))
    aqi_12_mean.append(np.mean(aqi_12))
    green_12_mean.append(np.mean(green_12))
    noise_12_mean.append(np.mean(noise_12))
    aqi_12_min.append(np.min(aqi_12))
    green_12_min.append(np.min(green_12))
    noise_12_min.append(np.min(noise_12))
    aqi_12_max.append(np.max(aqi_12))
    green_12_max.append(np.max(green_12))
    noise_12_max.append(np.max(noise_12))

paths = read_paths_from_file(inp13)
for path in paths:
    p=""
    for i in range(len(path)):
        p+=path[i]
    p=p.split(',')
    for i in range(len(p)):
        p[i]=int(p[i])
    src=p[0]
    dest=p[-1]
    rew1=13
    aqi_13,green_13,noise_13,r_13=calculatePathValues(p)
    r_13_mean.append(np.mean(r_13))
    aqi_13_mean.append(np.mean(aqi_13))
    green_13_mean.append(np.mean(green_13))
    noise_13_mean.append(np.mean(noise_13))
    aqi_13_min.append(np.min(aqi_13))
    green_13_min.append(np.min(green_13))
    noise_13_min.append(np.min(noise_13))
    aqi_13_max.append(np.max(aqi_13))
    green_13_max.append(np.max(green_13))
    noise_13_max.append(np.max(noise_13))

paths = read_paths_from_file(inp14)
for path in paths:
    p=""
    for i in range(len(path)):
        p+=path[i]
    p=p.split(',')
    for i in range(len(p)):
        p[i]=int(p[i])
    src=p[0]
    dest=p[-1]
    rew1=14
    aqi_14,green_14,noise_14,r_14=calculatePathValues(p)
    r_14_mean.append(np.mean(r_14))
    aqi_14_mean.append(np.mean(aqi_14))
    green_14_mean.append(np.mean(green_14))
    noise_14_mean.append(np.mean(noise_14))
    aqi_14_min.append(np.min(aqi_14))
    green_14_min.append(np.min(green_14))
    noise_14_min.append(np.min(noise_14))
    aqi_14_max.append(np.max(aqi_14))
    green_14_max.append(np.max(green_14))
    noise_14_max.append(np.max(noise_14))

model_1_mean_values={
    'aqi_mean': np.mean(aqi_1_mean),
    'green_mean': np.mean(green_1_mean),
    'noise_mean': np.mean(noise_1_mean),
}

model_1_min_values={
    'aqi_min': np.mean(aqi_1_min),
    'green_min': np.mean(green_1_min),
    'noise_min': np.mean(noise_1_min),
}

model_1_max_values={
    'aqi_max': np.mean(aqi_1_max),
    'green_max': np.mean(green_1_max),
    'noise_max': np.mean(noise_1_max),
}



model_2_mean_values={
    'aqi_mean': np.mean(aqi_2_mean),
    'green_mean': np.mean(green_2_mean),
    'noise_mean': np.mean(noise_2_mean),
}

model_2_min_values={
    'aqi_min': np.mean(aqi_2_min),
    'green_min': np.mean(green_2_min),
    'noise_min': np.mean(noise_2_min),
}

model_2_max_values={
    'aqi_max': np.mean(aqi_2_max),
    'green_max': np.mean(green_2_max),
    'noise_max': np.mean(noise_2_max),
}

model_3_mean_values={
    'aqi_mean': np.mean(aqi_3_mean),
    'green_mean': np.mean(green_3_mean),
    'noise_mean': np.mean(noise_3_mean),
}

model_3_min_values={
    'aqi_min': np.mean(aqi_3_min),
    'green_min': np.mean(green_3_min),
    'noise_min': np.mean(noise_3_min),
}

model_3_max_values={
    'aqi_max': np.mean(aqi_3_max),
    'green_max': np.mean(green_3_max),
    'noise_max': np.mean(noise_3_max),
}


model_4_mean_values={
    'aqi_mean': np.mean(aqi_4_mean),
    'green_mean': np.mean(green_4_mean),
    'noise_mean': np.mean(noise_4_mean),
}

model_4_min_values={
    'aqi_min': np.mean(aqi_4_min),
    'green_min': np.mean(green_4_min),
    'noise_min': np.mean(noise_4_min),
}

model_4_max_values={
    'aqi_max': np.mean(aqi_4_max),
    'green_max': np.mean(green_4_max),
    'noise_max': np.mean(noise_4_max),
}

model_5_mean_values={
    'aqi_mean': np.mean(aqi_5_mean),
    'green_mean': np.mean(green_5_mean),
    'noise_mean': np.mean(noise_5_mean),
}

model_5_min_values={
    'aqi_min': np.mean(aqi_5_min),
    'green_min': np.mean(green_5_min),
    'noise_min': np.mean(noise_5_min),
}

model_5_max_values={
    'aqi_max': np.mean(aqi_5_max),
    'green_max': np.mean(green_5_max),
    'noise_max': np.mean(noise_5_max),
}

model_6_mean_values={
    'aqi_mean': np.mean(aqi_6_mean),
    'green_mean': np.mean(green_6_mean),
    'noise_mean': np.mean(noise_6_mean),
}

model_6_min_values={
    'aqi_min': np.mean(aqi_6_min),
    'green_min': np.mean(green_6_min),
    'noise_min': np.mean(noise_6_min),
}

model_6_max_values={
    'aqi_max': np.mean(aqi_6_max),
    'green_max': np.mean(green_6_max),
    'noise_max': np.mean(noise_6_max),
}

model_7_mean_values={
    'aqi_mean': np.mean(aqi_7_mean),
    'green_mean': np.mean(green_7_mean),
    'noise_mean': np.mean(noise_7_mean),
}

model_7_min_values={
    'aqi_min': np.mean(aqi_7_min),
    'green_min': np.mean(green_7_min),
    'noise_min': np.mean(noise_7_min),
}

model_7_max_values={
    'aqi_max': np.mean(aqi_7_max),
    'green_max': np.mean(green_7_max),
    'noise_max': np.mean(noise_7_max),
}

model_8_mean_values={
    'aqi_mean': np.mean(aqi_8_mean),
    'green_mean': np.mean(green_8_mean),
    'noise_mean': np.mean(noise_8_mean),
}

model_8_min_values={
    'aqi_min': np.mean(aqi_8_min),
    'green_min': np.mean(green_8_min),
    'noise_min': np.mean(noise_8_min),
}

model_8_max_values={
    'aqi_max': np.mean(aqi_8_max),
    'green_max': np.mean(green_8_max),
    'noise_max': np.mean(noise_8_max),
}

model_9_mean_values={
    'aqi_mean': np.mean(aqi_9_mean),
    'green_mean': np.mean(green_9_mean),
    'noise_mean': np.mean(noise_9_mean),
}

model_9_min_values={
    'aqi_min': np.mean(aqi_9_min),
    'green_min': np.mean(green_9_min),
    'noise_min': np.mean(noise_9_min),
}

model_9_max_values={
    'aqi_max': np.mean(aqi_9_max),
    'green_max': np.mean(green_9_max),
    'noise_max': np.mean(noise_9_max),
}

model_10_mean_values={
    'aqi_mean': np.mean(aqi_10_mean),
    'green_mean': np.mean(green_10_mean),
    'noise_mean': np.mean(noise_10_mean),
}

model_10_min_values={
    'aqi_min': np.mean(aqi_10_min),
    'green_min': np.mean(green_10_min),
    'noise_min': np.mean(noise_10_min),
}

model_10_max_values={
    'aqi_max': np.mean(aqi_10_max),
    'green_max': np.mean(green_10_max),
    'noise_max': np.mean(noise_10_max),
}

model_11_mean_values={
    'aqi_mean': np.mean(aqi_11_mean),
    'green_mean': np.mean(green_11_mean),
    'noise_mean': np.mean(noise_11_mean),
}

model_11_min_values={
    'aqi_min': np.mean(aqi_11_min),
    'green_min': np.mean(green_11_min),
    'noise_min': np.mean(noise_11_min),
}

model_11_max_values={
    'aqi_max': np.mean(aqi_11_max),
    'green_max': np.mean(green_11_max),
    'noise_max': np.mean(noise_11_max),
}

model_12_mean_values={
    'aqi_mean': np.mean(aqi_12_mean),
    'green_mean': np.mean(green_12_mean),
    'noise_mean': np.mean(noise_12_mean),
}

model_12_min_values={
    'aqi_min': np.mean(aqi_12_min),
    'green_min': np.mean(green_12_min),
    'noise_min': np.mean(noise_12_min),
}

model_12_max_values={
    'aqi_max': np.mean(aqi_12_max),
    'green_max': np.mean(green_12_max),
    'noise_max': np.mean(noise_12_max),
}

model_13_mean_values={
    'aqi_mean': np.mean(aqi_13_mean),
    'green_mean': np.mean(green_13_mean),
    'noise_mean': np.mean(noise_13_mean),
}

model_13_min_values={
    'aqi_min': np.mean(aqi_13_min),
    'green_min': np.mean(green_13_min),
    'noise_min': np.mean(noise_13_min),
}

model_13_max_values={
    'aqi_max': np.mean(aqi_13_max),
    'green_max': np.mean(green_13_max),
    'noise_max': np.mean(noise_13_max),
}

model_14_mean_values={
    'aqi_mean': np.mean(aqi_14_mean),
    'green_mean': np.mean(green_14_mean),
    'noise_mean': np.mean(noise_14_mean),
}

model_14_min_values={
    'aqi_min': np.mean(aqi_14_min),
    'green_min': np.mean(green_14_min),
    'noise_min': np.mean(noise_14_min),
}

model_14_max_values={
    'aqi_max': np.mean(aqi_14_max),
    'green_max': np.mean(green_14_max),
    'noise_max': np.mean(noise_14_max),
}



metrics = list(model_1_mean_values.keys())
metric_labels = ['AQI Mean', 'Green Mean', 'Noise Mean']


# Set the width of the bars
bar_width = 0.1

# Set the positions of the bars on the x-axis
r1 = np.arange(len(metrics))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]
r5 = [x + bar_width for x in r4]
r6 = [x + bar_width for x in r5]
r7 = [x + bar_width for x in r6]
r8 = [x + bar_width for x in r7]
r9 = [x + bar_width for x in r8]
r10 = [x + bar_width for x in r9]
r11 = [x + bar_width for x in r10]
r12 = [x + bar_width for x in r11]
r13 = [x + bar_width for x in r12]
r14 = [x + bar_width for x in r13]


model_1_mean_values = [model_1_mean_values[metric] for metric in metrics]
model_2_mean_values = [model_2_mean_values[metric] for metric in metrics]
model_3_mean_values = [model_3_mean_values[metric] for metric in metrics]
model_4_mean_values = [model_4_mean_values[metric] for metric in metrics]
model_5_mean_values = [model_5_mean_values[metric] for metric in metrics]
model_6_mean_values = [model_6_mean_values[metric] for metric in metrics]
model_7_mean_values = [model_7_mean_values[metric] for metric in metrics]
model_8_mean_values = [model_8_mean_values[metric] for metric in metrics]
model_9_mean_values = [model_9_mean_values[metric] for metric in metrics]
model_10_mean_values = [model_10_mean_values[metric] for metric in metrics]
model_11_mean_values = [model_11_mean_values[metric] for metric in metrics]
model_12_mean_values = [model_12_mean_values[metric] for metric in metrics]
model_13_mean_values = [model_13_mean_values[metric] for metric in metrics]
model_14_mean_values = [model_14_mean_values[metric] for metric in metrics]

# print(model_1_mean_values,model_2_mean_values,model_4_mean_values)


# Plotting the bars
# plt.figure(figsize=(12, 8))
# plt.bar(r1,model_1_mean_values, color='blue', width=bar_width, edgecolor='grey', label='gama=.9')
# plt.bar(r2,model_2_mean_values, color='green', width=bar_width, edgecolor='grey', label='gama=.8')
# plt.bar(r3, model_4_mean_values, color='red', width=bar_width, edgecolor='grey', label='gama=.5')
# plt.bar(r4, model_6_mean_values, color='yellow', width=bar_width, edgecolor='grey', label='gama=.1')
# plt.bar(r5, model_3_mean_values, color='black', width=bar_width, edgecolor='grey', label='gama=.9,w=.001')
# plt.bar(r6, model_5_mean_values, color='purple', width=bar_width, edgecolor='grey', label='gama=.5,w=.001')
# plt.bar(r7, model_7_mean_values, color='orange', width=bar_width, edgecolor='grey', label='gama=.9,w=.1')
# plt.bar(r8, model_8_mean_values, color='pink', width=bar_width, edgecolor='grey', label='gama=.9,w=.000001')
# plt.bar(r9, model_9_mean_values, color='brown', width=bar_width, edgecolor='grey', label='gama=.1,w=.001')


# # Adding labels
# plt.xlabel('Metrics', fontweight='bold')
# plt.xticks([r + bar_width for r in range(len(metrics))], metric_labels)
# plt.ylabel('Mean Value (Log Scale)', fontweight='bold')
# plt.yscale('log')  # Set the y-axis to a logarithmic scale
# plt.title('Comparison of Metrics for Different Path Types')
# plt.legend()

# # Show plot
# plt.savefig(dt+'metric_comparison_mean.png')
# plt.close()


# metrics = list(model_1_min_values.keys())
# metric_labels = ['AQI Min', 'Green Min', 'Noise Min']



# model_1_min_values = [model_1_min_values[metric] for metric in metrics]
# model_2_min_values = [model_2_min_values[metric] for metric in metrics]
# model_3_min_values = [model_3_min_values[metric] for metric in metrics]
# model_4_min_values = [model_4_min_values[metric] for metric in metrics]
# model_5_min_values = [model_5_min_values[metric] for metric in metrics]
# model_6_min_values = [model_6_min_values[metric] for metric in metrics]
# model_7_min_values = [model_7_min_values[metric] for metric in metrics]
# model_8_min_values = [model_8_min_values[metric] for metric in metrics]
# model_9_min_values = [model_9_min_values[metric] for metric in metrics]



# # Plotting the bars
# plt.figure(figsize=(12, 8))
# plt.bar(r1,model_1_min_values, color='blue', width=bar_width, edgecolor='grey', label='gama=.9')
# plt.bar(r2,model_2_min_values, color='green', width=bar_width, edgecolor='grey', label='gama=.8')
# plt.bar(r3, model_4_min_values, color='red', width=bar_width, edgecolor='grey', label='gama=.5')
# plt.bar(r4, model_6_min_values, color='yellow', width=bar_width, edgecolor='grey', label='gama=.1')
# plt.bar(r5, model_3_min_values, color='black', width=bar_width, edgecolor='grey', label='gama=.9,w=.001')
# plt.bar(r6, model_5_min_values, color='purple', width=bar_width, edgecolor='grey', label='gama=.5,w=.001')
# plt.bar(r7, model_7_min_values, color='orange', width=bar_width, edgecolor='grey', label='gama=.9,w=.1')
# plt.bar(r8, model_8_min_values, color='pink', width=bar_width, edgecolor='grey', label='gama=.9,w=.000001')
# plt.bar(r9, model_9_min_values, color='brown', width=bar_width, edgecolor='grey', label='gama=.1,w=.001')

# # Adding labels
# plt.xlabel('Metrics', fontweight='bold')
# plt.xticks([r + bar_width for r in range(len(metrics))], metric_labels)
# plt.ylabel('Min Value (Log Scale)', fontweight='bold')
# plt.yscale('log')  # Set the y-axis to a logarithmic scale
# plt.title('Comparison of Metrics for Different Path Types')
# plt.legend()

# # Show plot
# plt.savefig(dt+'metric_comparison_min.png')
# plt.close()

# metrics = list(model_1_max_values.keys())
# metric_labels = ['AQI Max', 'Green Max', 'Noise Max']


# model_1_max_values = [model_1_max_values[metric] for metric in metrics]
# model_2_max_values = [model_2_max_values[metric] for metric in metrics]
# model_3_max_values = [model_3_max_values[metric] for metric in metrics]
# model_4_max_values = [model_4_max_values[metric] for metric in metrics]
# model_5_max_values = [model_5_max_values[metric] for metric in metrics]
# model_6_max_values = [model_6_max_values[metric] for metric in metrics]
# model_7_max_values = [model_7_max_values[metric] for metric in metrics]
# model_8_max_values = [model_8_max_values[metric] for metric in metrics]
# model_9_max_values = [model_9_max_values[metric] for metric in metrics]

# plt.figure(figsize=(12, 8))
# plt.bar(r1,model_1_max_values, color='blue', width=bar_width, edgecolor='grey', label='gama=.9')
# plt.bar(r2,model_2_max_values, color='green', width=bar_width, edgecolor='grey', label='gama=.8')
# plt.bar(r3, model_4_max_values, color='red', width=bar_width, edgecolor='grey', label='gama=.5')
# plt.bar(r4, model_6_max_values, color='yellow', width=bar_width, edgecolor='grey', label='gama=.1')
# plt.bar(r5, model_3_max_values, color='black', width=bar_width, edgecolor='grey', label='gama=.9,w=.001')
# plt.bar(r6, model_5_max_values, color='purple', width=bar_width, edgecolor='grey', label='gama=.5,w=.001')
# plt.bar(r7, model_7_max_values, color='orange', width=bar_width, edgecolor='grey', label='gama=.9,w=.1')
# plt.bar(r8, model_8_max_values, color='pink', width=bar_width, edgecolor='grey', label='gama=.9,w=.000001')
# plt.bar(r9, model_9_max_values, color='brown', width=bar_width, edgecolor='grey', label='gama=.1,w=.001')

# plt.xlabel('Metrics', fontweight='bold')
# plt.xticks([r + bar_width for r in range(len(metrics))], metric_labels)
# plt.ylabel('Max Value (Log Scale)', fontweight='bold')
# plt.yscale('log')  # Set the y-axis to a logarithmic scale
# plt.title('Comparison of Metrics for Different Path Types')
# plt.legend()

# plt.savefig(dt+'metric_comparison_max.png')
# plt.close()


models = ['M1', 'M2', 'M3', 'M4','M5','M6','M7','M8','M9','M10','M11','M12','M13','M14']
values = [np.mean(r_1_mean), np.mean(r_2_mean), np.mean(r_3_mean), np.mean(r_4_mean),np.mean(r_5_mean),np.mean(r_6_mean),np.mean(r_7_mean),np.mean(r_8_mean),np.mean(r_9_mean),np.mean(r_10_mean),np.mean(r_11_mean),np.mean(r_12_mean),np.mean(r_13_mean),np.mean(r_14_mean)]
lables=['gamma=.9','gamma=.8','gamma=.9,w=.001','gamma=.5','gamma=.5,w=.001','gamma=.1','gamma=.9,w=.1','gamma=.9,w=.000001','gamma=.1,w=.001','gamma=.4','g=.4,w=.000000001','lr=.005','lr=.0015','lr=.00075']

colors = ['blue', 'green', 'black', 'red','purple','yellow','orange','pink','brown','grey','cyan','magenta','brown','black']
# Plotting
plt.figure(figsize=(6, 4))
for i in range(len(models)):
    plt.bar(models[i], values[i], color=colors[i], label=lables[i])


# Labels, title
plt.xlabel('Models')
plt.ylabel('Average Reward (Log Scale)', fontweight='bold')
plt.yscale('log')  # Set the y-axis to a logarithmic scale
plt.title('Comparison of Metrics for Different Path Types')
plt.legend()



plt.savefig(dt+'av_reward.png')
plt.close()


# Sample data for the three subplots
models1 = ['gamma=.7', 'gamma=.5', 'gamma=.4', 'gamma=.2']
values1 = [np.mean(r_2_mean) / 10, np.mean(r_4_mean) / 10, np.mean(r_10_mean) / 10, np.mean(r_6_mean) / 10]

models2 = ['w=.001', 'w=.1', 'w=.000001', 'w=.0000001']
values2 = [np.mean(r_3_mean) / 10, np.mean(r_7_mean) / 10, np.mean(r_8_mean) / 10, np.mean(r_11_mean) / 10]

models3 = ['lr=.0005', 'lr=.00075', 'lr=.001', 'lr=.0015']
values3 = [np.mean(r_12_mean) / 10, np.mean(r_14_mean) / 10, np.mean(r_11_mean)/10, np.mean(r_13_mean) / 10]

bar_color = '#4682B4'
bar_width = 0.45

# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot the first subplot
axes[0].bar(models1, values1, width=bar_width, color=bar_color)
axes[0].set_xlabel('Discount Factor', fontsize=18)
axes[0].set_ylabel('Average Reward', fontsize=18)
axes[0].set_title('Effect of Varying Discount Factor', fontsize=18)
axes[0].set_yscale('log')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))
axes[0].yaxis.set_minor_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))
axes[0].tick_params(axis='x', labelsize=12)
axes[0].tick_params(axis='y', labelsize=20)

# Plot the second subplot
axes[1].bar(models2, values2, width=bar_width, color=bar_color)
axes[1].set_xlabel('Weight in Reward', fontsize=18)
axes[1].set_ylabel('Average Reward', fontsize=18)
axes[1].set_title('Effect of Varying Weight in Reward', fontsize=18)
axes[1].set_yscale('log')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))
axes[1].yaxis.set_minor_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))
axes[1].tick_params(axis='x', labelsize=12)
axes[1].tick_params(axis='y', labelsize=20)

# Plot the third subplot
axes[2].bar(models3, values3, width=bar_width, color=bar_color)
axes[2].set_xlabel('Learning Rate', fontsize=18)
axes[2].set_ylabel('Average Reward', fontsize=18)
axes[2].set_title('Effect of Varying Learning Rate', fontsize=18)
axes[2].set_yscale('log')
axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))
axes[2].yaxis.set_minor_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))
axes[2].tick_params(axis='x', labelsize=14)
axes[2].tick_params(axis='y', labelsize=20)

# Adjust the layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig('combined_figure.png')
plt.close()
