import ast
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import random
import networkx as nx
import math
inp='exp16_train.csv'
dd='exp16/train/'
dt='exp16/train/'
dtt='exp16/train'

# Function to read paths from file
def random_path_nx(G,source,target):
    paths=nx.all_simple_paths(G,source,target,cutoff=10)
    return random.choice(list(paths))

def create_graph(green_path, shortest_path, random_path, x_values, title_name, fname, n):
    # Make sure all arrays have the same length as the shortest array among them
    min_length = min(len(green_path), len(shortest_path), len(random_path), len(x_values))
    green_path = green_path[:min_length]
    shortest_path = shortest_path[:min_length]
    random_path = random_path[:min_length]
    x_values = x_values[:min_length]

    # Determine the number of segments and the size of each segment
    num_segments = n
    segment_size = min_length // num_segments

    # Loop through each segment and create a graph
    for segment_index in range(num_segments):
        # Calculate start and end indices for the current segment
        start_index = segment_index * segment_size
        end_index = start_index + segment_size

        # Extract the current segment for x_values and each path
        segment_x_values = x_values[start_index:end_index]
        segment_green_path = green_path[start_index:end_index]
        segment_shortest_path = shortest_path[start_index:end_index]
        segment_random_path = random_path[start_index:end_index]

        # Create the plot for the current segment
        plt.figure(figsize=(10, 6))
        plt.plot(segment_x_values, segment_green_path, label='Green Path', color='red', marker='o')
        plt.plot(segment_x_values, segment_shortest_path, label='Shortest Path', color='blue', marker='o')
        plt.plot(segment_x_values, segment_random_path, label='Random Path', color='green', marker='o')

        plt.xlabel('Episode Number')
        plt.ylabel(f'{title_name} Value')
        plt.title(f'{title_name} Values for Segment {segment_index + 1}')
        plt.grid(True)
        plt.legend()

        # Construct the filename for the current segment
        filename = f'{fname}_segment_{segment_index + 1}.png'

        # Ensure the 'result_fr' directory exists
        # if not os.path.exists('exp3'):
        #     os.makedirs('exp3')

        # Save the plot
        plt.savefig(os.path.join(dtt, filename))

        # Clear the figure to avoid overlap with subsequent plots
        plt.close()


def draw_comparison_chart(values_greenpath,values_randompath,values_dijkstra , column_labels,tye):


    # Extract values for specified columns from each dataset


    # Number of groups
    n_groups = len(values_greenpath)

    # Create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.25

    # Bars for each path type within each column group
    rects1 = ax.bar(index, values_greenpath, bar_width, label='Green Path')
    rects2 = ax.bar(index + bar_width, values_randompath, bar_width, label='Random Path')
    rects3 = ax.bar(index + 2 * bar_width, values_dijkstra, bar_width, label='Dijkstra')

    # Adding labels, title, and axes ticks
    ax.set_xlabel('Metric')
    ax.set_ylabel('Values')
    ax.set_title('Metric Comparison by Path Type')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(column_labels)
    ax.legend()

    # convert column_labels into a single string to represent the filename
    filename = '_'.join(column_labels).replace(' ', '_').lower()

    # Showing the plot
    plt.tight_layout()
    plt.savefig(dt + filename +'_'+tye+ '.png')

rew1=0
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
    w=.000001
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

    # calculate total length from path[0] to path[-1]
    for i in range(len(pathArr)-1):
        total_len+= G[pathArr[i]][pathArr[i+1]]['e_l']

    for i in range(len(pathArr)-1):
        aqi_random.append( G[pathArr[i]][pathArr[i+1]]['aqi'])
        green_random.append( G[pathArr[i]][pathArr[i+1]]['e_g'])
        noise_random.append( G[pathArr[i]][pathArr[i+1]]['e_n'])

        total_reward.append(calculate_combined_reward2(G[pathArr[i]][pathArr[i+1]]['aqi'],G[pathArr[i]][pathArr[i+1]]['e_g'],G[pathArr[i]][pathArr[i+1]]['e_n'],sp,total_len))

    return aqi_random,green_random,noise_random,total_reward


def read_paths_from_file(file_path):
    with open(file_path, 'r') as file:
        # Read the entire content of the file as a single string
        content = file.readlines()

    return content

# Assuming your file is named 'paths.txt' and located in the current directory

paths = read_paths_from_file(inp)

G = pd.read_pickle("normalized_graph.gpickle")
reverse_G = pd.read_pickle("reversed_normalized_graph.gpickle")  



episode=0
aqi_rl_mean=[]
green_rl_mean=[]
noise_rl_mean=[]
aqi_random_mean=[]
green_random_mean=[]
noise_random_mean=[]
aqi_shortest_mean=[]
green_shortest_mean=[]
noise_shortest_mean=[]
aqi_rl_min=[]
green_rl_min=[]
noise_rl_min=[]
aqi_random_min=[]
green_random_min=[]
noise_random_min=[]
aqi_shortest_min=[]
green_shortest_min=[]
noise_shortest_min=[]
aqi_rl_max=[]
green_rl_max=[]
noise_rl_max=[]
aqi_random_max=[]
green_random_max=[]
noise_random_max=[]
aqi_shortest_max=[]
green_shortest_max=[]
noise_shortest_max=[]
cnt=0
reward_random=[]
reward_rl=[]
reward_shortest=[]
for path in paths:
    p=""
    for i in range(len(path)):
        p+=path[i]
    p=p.split(',')
    for i in range(len(p)):
        p[i]=int(p[i])
    src=p[0]
    dest=p[-1]
    # print('src: ',src,'dest: ',dest)
    r_path=random_path_nx(G,src,dest)
    # print("r Path: ",r_path)
    aqi_random,green_random,noise_random,total_reward=calculatePathValues(r_path)
    aqi_random_mean.append(np.mean(aqi_random))
    green_random_mean.append(np.mean(green_random))
    noise_random_mean.append(np.mean(noise_random))
    aqi_random_min.append(np.min(aqi_random))
    green_random_min.append(np.min(green_random))
    noise_random_min.append(np.min(noise_random))
    aqi_random_max.append(np.max(aqi_random))
    green_random_max.append(np.max(green_random))
    noise_random_max.append(np.max(noise_random))
    reward_random.append(np.mean(total_reward))
    
    aqi_green,green_green,noise_green,rew_rl=calculatePathValues(p)
    
    aqi_rl_mean.append(np.mean(aqi_green))
    green_rl_mean.append(np.mean(green_green))
    noise_rl_mean.append(np.mean(noise_green))
    aqi_rl_min.append(np.min(aqi_green))
    green_rl_min.append(np.min(green_green))
    noise_rl_min.append(np.min(noise_green))
    aqi_rl_max.append(np.max(aqi_green))
    green_rl_max.append(np.max(green_green))
    noise_rl_max.append(np.max(noise_green))
    reward_rl.append(np.mean(rew_rl))

    aqi_shortest,green_shortest,noise_shortest,rew_short=calculatePathValues(nx.shortest_path(G,src,dest))
    aqi_shortest_mean.append(np.mean(aqi_shortest))
    green_shortest_mean.append(np.mean(green_shortest))
    noise_shortest_mean.append(np.mean(noise_shortest))
    aqi_shortest_min.append(np.min(aqi_shortest))
    green_shortest_min.append(np.min(green_shortest))
    noise_shortest_min.append(np.min(noise_shortest))
    aqi_shortest_max.append(np.max(aqi_shortest))
    green_shortest_max.append(np.max(green_shortest))
    noise_shortest_max.append(np.max(noise_shortest))
    reward_shortest.append(np.mean(rew_short))

# create_graph(green_rl_mean, green_shortest_mean, green_random_mean, range(len(green_rl_mean)),'GVI Mean Train','gvi_mean_train',5)  
# create_graph(aqi_rl_mean, aqi_shortest_mean, aqi_random_mean, range(len(aqi_rl_mean)),'AQI Mean Train','aqi_mean_train',5)  
# create_graph(noise_rl_mean, noise_shortest_mean, noise_random_mean, range(len(noise_rl_mean)),'Noise Mean Train','noise_mean_train',5)
# create_graph(green_rl_min, green_shortest_min, green_random_min, range(len(green_rl_min)),'GVI Min Train','gvi_min_train',5)
# create_graph(aqi_rl_min, aqi_shortest_min, aqi_random_min, range(len(aqi_rl_min)),'AQI Min Train','aqi_min_train',5)
# create_graph(noise_rl_min, noise_shortest_min, noise_random_min, range(len(noise_rl_min)),'Noise Min Train','noise_min_train',5)
# create_graph(green_rl_max, green_shortest_max, green_random_max, range(len(green_rl_max)),'GVI Max Train','gvi_max_train',5)
# create_graph(aqi_rl_max, aqi_shortest_max, aqi_random_max, range(len(aqi_rl_max)),'AQI Max Train','aqi_max_train',5)
# create_graph(noise_rl_max, noise_shortest_max, noise_random_max, range(len(noise_rl_max)),'Noise Max Train','noise_max_train',5)

print("Mean green_rl")
print(np.mean(aqi_rl_mean),np.mean(green_rl_mean),np.mean(noise_rl_mean))
print("Mean green_random")
print(np.mean(aqi_random_mean),np.mean(green_random_mean),np.mean(noise_random_mean))
print("Mean green_shortest")
print(np.mean(aqi_shortest_mean),np.mean(green_shortest_mean),np.mean(noise_shortest_mean))
print("Min green_rl")
print(np.mean(aqi_rl_min),np.mean(green_rl_min),np.mean(noise_rl_min))
print("Min green_random")
print(np.mean(aqi_random_min),np.mean(green_random_min),np.mean(noise_random_min))
print("Min green_shortest")
print(np.mean(aqi_shortest_min),np.mean(green_shortest_min),np.mean(noise_shortest_min))
print("Max green_rl")
print(np.mean(aqi_rl_max),np.mean(green_rl_max),np.mean(noise_rl_max))
print("Max green_random")
print(np.mean(aqi_random_max),np.mean(green_random_max),np.mean(noise_random_max))
print("Max green_shortest")
print(np.mean(aqi_shortest_max),np.mean(green_shortest_max),np.mean(noise_shortest_max))
print("Mean reward_rl")
print(np.mean(reward_rl))
print("Mean reward_random")
print(np.mean(reward_random))
print("Mean reward_shortest")
print(np.mean(reward_shortest))


print("cnt: ",cnt)

model_mean_values={
    'aqi_mean': np.mean(aqi_rl_mean),
    'green_mean': np.mean(green_rl_mean),
    'noise_mean': np.mean(noise_rl_min),
    'aqi_min': np.mean(aqi_rl_min),
    'green_min': np.mean(green_rl_min),
    'noise_min': np.mean(noise_rl_min),
    'aqi_max': np.mean(aqi_rl_max),
    'green_max': np.mean(green_rl_max),
    'noise_max': np.mean(noise_rl_max)
}

random_means_values={
    'aqi_mean': np.mean(aqi_random_mean),
    'green_mean': np.mean(green_random_mean),
    'noise_mean': np.mean(noise_random_mean),
    'aqi_min': np.mean(aqi_random_min),
    'green_min': np.mean(green_random_min),
    'noise_min': np.mean(noise_random_min),
    'aqi_max': np.mean(aqi_random_max),
    'green_max': np.mean(green_random_max),
    'noise_max': np.mean(noise_random_max)
}

shortest_mean_values = {
    'aqi_mean': np.mean(aqi_shortest_mean),
    'green_mean': np.mean(green_shortest_mean),
    'noise_mean': np.mean(noise_shortest_mean),
    'aqi_min': np.mean(aqi_shortest_min),
    'green_min': np.mean(green_shortest_min),
    'noise_min': np.mean(noise_shortest_min),
    'aqi_max': np.mean(aqi_shortest_max),
    'green_max': np.mean(green_shortest_max),
    'noise_max': np.mean(noise_shortest_max)
}

metrics = list(model_mean_values.keys())
metric_labels = ['AQI Mean', 'GVI Mean', 'Noise Mean', 'AQI Min', 'GVI Min', 'Noise Min', 'AQI Max', 'GVI Max', 'Noise Max']
model_mean_values = [model_mean_values[metric] for metric in metrics]
random_means_values = [random_means_values[metric] for metric in metrics]
shortest_mean_values = [shortest_mean_values[metric] for metric in metrics]

# Set the width of the bars
bar_width = 0.15
mean_indices = [0, 1, 2]  # AQI Mean, Green Mean, Noise Mean
min_indices = [3, 4, 5]   # Min AQI, Min Green, Min Noise
max_indices = [6, 7, 8]   # Max AQI, Max Green, Max Noise

# Define a function to plot each category
def plot_metrics(indices, title, ylabel, fig_name):
    metrics_subset = [metrics[i] for i in indices]
    metric_labels_subset = [metric_labels[i] for i in indices]
    model_values_subset = [model_mean_values[i] for i in indices]
    random_values_subset = [random_means_values[i] for i in indices]
    shortest_values_subset = [shortest_mean_values[i] for i in indices]
    
    r = np.arange(len(metrics_subset))
    r2 = [x + bar_width for x in r]
    r3 = [x + bar_width for x in r2]

    plt.figure(figsize=(6,4))
    plt.bar(r, model_values_subset, color='green', width=bar_width, edgecolor='grey', label='Green Route')
    plt.bar(r2, random_values_subset, color='blue', width=bar_width, edgecolor='grey', label='Random Route')
    plt.bar(r3, shortest_values_subset, color='red', width=bar_width, edgecolor='grey', label='Shortest Route')
    
    # plt.xlabel('Metrics', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(metrics_subset))], metric_labels_subset)
    # plt.ylabel(ylabel, fontweight='bold')
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))
    plt.gca().yaxis.set_minor_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))
    plt.title(title)
    plt.legend()
    
    plt.savefig(fig_name)
    plt.close()

# Plotting
plot_metrics(mean_indices, 'Comparison of Mean Values for Different Routes', 'Mean Value', 'mean_comparison.png')
plot_metrics(min_indices, 'Comparison of Minimum Values for Different Routes', 'Minimum Value', 'min_comparison.png')
plot_metrics(max_indices, 'Comparison of Maximum Values for Different Routes', 'Maximum Value', 'max_comparison.png')



# Group the values from other models for easier iteration
other_models_values = [shortest_mean_values, random_means_values]
other_models_names = ['Shortest', 'Random',]

# Function to calculate percentage change
def calculate_percentage_change(your_model_value, other_model_value):
    if your_model_value == 0:  # Prevent division by zero
        return float('inf')  # Indicate undefined percentage change
    return (( your_model_value-other_model_value ) / your_model_value) * 100

# Iterate over each metric and calculate the percentage change compared to other models
for metric_index, metric in enumerate(metrics):
    print(f"Metric: {metric}")
    for model_index, other_model_values in enumerate(other_models_values):
        percentage_change = calculate_percentage_change(model_mean_values[metric_index], other_model_values[metric_index])
        change_type = "increase" if percentage_change >= 0 else "decrease"
        print(f" - From {other_models_names[model_index]} model, my model shows a {abs(percentage_change):.2f}% {change_type} for {metric}.")
    print()  # Add a blank line for better readability

average_rewards = {
    'Green Route': np.mean(reward_random),  # Replace with the actual average reward for Green Route
    'Shortest Route': np.mean(reward_shortest),  # Replace with the actual average reward for Shortest Route
    'Random Route': np.mean(reward_rl)  # Replace with the actual average reward for Random Route
}

routes = list(average_rewards.keys())
rewards = list(average_rewards.values())

r = np.arange(len(routes))
bar_width = 0.25  # Define a suitable bar width

# Create figure and axes
plt.figure(figsize=(6,4))  # Increased figure size

# Plotting bars for each route type with colors and labels
plt.bar(r, rewards, color=['green', 'blue', 'red'], width=bar_width, edgecolor='grey')


plt.xticks(r, routes)
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))
plt.gca().yaxis.set_minor_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))

# Adding plot details
plt.title('Comparison of Average Rewards for Different Routes')
# plt.xlabel('Route Types')
plt.ylabel('Average Reward')



# Ensure layout fits well
plt.tight_layout()

# Save the figure to a file
plt.savefig('average_rewards_comparison.png')
plt.close()

