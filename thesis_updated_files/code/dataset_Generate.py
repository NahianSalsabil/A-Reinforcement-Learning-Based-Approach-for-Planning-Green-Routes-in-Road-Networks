import random
import networkx as nx
import pickle
import pandas as pd

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

graph=pd.read_pickle('full_graph.gpickle')



dataset_1000_5=(generate_random_dataset(graph,5,1000))
dataset_2000_5=(generate_random_dataset(graph,5,2000))
dataset_3000_5=(generate_random_dataset(graph,5,2000))

dataset_1000_10=(generate_random_dataset(graph,10,1000))
dataset_2000_10=(generate_random_dataset(graph,10,2000))
dataset_3000_10=(generate_random_dataset(graph,10,2000))


with open('mixed_1000_5.txt','w') as file:
    for row in dataset_1000_5:
        file.write(row[0]+','+row[1]+'\n')

with open('mixed_2000_5.txt','w') as file:
    for row in dataset_2000_5:
        file.write(row[0]+','+row[1]+'\n')

with open('mixed_3000_5.txt','w') as file:
    for row in dataset_3000_5:
        file.write(row[0]+','+row[1]+'\n')

with open('mixed_1000_10.txt','w') as file:
    for row in dataset_1000_10:
        file.write(row[0]+','+row[1]+'\n')

with open('mixed_2000_10.txt','w') as file:
    for row in dataset_2000_10:
        file.write(row[0]+','+row[1]+'\n')

with open('mixed_3000_10.txt','w') as file:
    for row in dataset_3000_10:
        file.write(row[0]+','+row[1]+'\n')

# all 5 hop

with open('mixed_1000_5.txt', 'r') as input_file:
    lines = input_file.readlines()

# Randomly sample 500 lines
sampled_lines = random.sample(lines, 700)

# Remove sampled lines from the original list
lines = [line for line in lines if line not in sampled_lines]

# Write the sampled lines to the first output file
with open('train_data_1000_5.txt', 'w') as output_file_1:
    output_file_1.writelines(sampled_lines)

# Randomly sample 200 lines from the remaining lines
sampled_lines_2 = random.sample(lines, 200)

# Write the second set of sampled lines to the second output file
with open('validation_data_1000_5.txt', 'w') as output_file_2:
    output_file_2.writelines(sampled_lines_2)
lines2 = [line for line in lines if line not in sampled_lines and line not in sampled_lines_2]

sampled_lines_3 = random.sample(lines2, 100)
with open('test_data_1000_5.txt', 'w') as output_file_3:
    output_file_3.writelines(sampled_lines_3)


with open('mixed_2000_5.txt', 'r') as input_file:
    lines = input_file.readlines()

# Randomly sample 500 lines
sampled_lines = random.sample(lines, 1400)

# Remove sampled lines from the original list
lines = [line for line in lines if line not in sampled_lines]

# Write the sampled lines to the first output file
with open('train_data_2000_5.txt', 'w') as output_file_1:
    output_file_1.writelines(sampled_lines)

# Randomly sample 200 lines from the remaining lines
sampled_lines_2 = random.sample(lines, 400)

# Write the second set of sampled lines to the second output file
with open('validation_data_2000_5.txt', 'w') as output_file_2:
    output_file_2.writelines(sampled_lines_2)
lines2 = [line for line in lines if line not in sampled_lines and line not in sampled_lines_2]

sampled_lines_3 = random.sample(lines2, 200)
with open('test_data_2000_5.txt', 'w') as output_file_3:
    output_file_3.writelines(sampled_lines_3)

with open('mixed_3000_5.txt', 'r') as input_file:
    lines = input_file.readlines()

# Randomly sample 500 lines
sampled_lines = random.sample(lines, 2100)

# Remove sampled lines from the original list
lines = [line for line in lines if line not in sampled_lines]

# Write the sampled lines to the first output file
with open('train_data_3000_5.txt', 'w') as output_file_1:
    output_file_1.writelines(sampled_lines)

# Randomly sample 200 lines from the remaining lines
sampled_lines_2 = random.sample(lines, 600)

# Write the second set of sampled lines to the second output file
with open('validation_data_3000_5.txt', 'w') as output_file_2:
    output_file_2.writelines(sampled_lines_2)
lines2 = [line for line in lines if line not in sampled_lines and line not in sampled_lines_2]

sampled_lines_3 = random.sample(lines2, 300)
with open('test_data_3000_5.txt', 'w') as output_file_3:
    output_file_3.writelines(sampled_lines_3)


# all 10 hop
    
with open('mixed_1000_10.txt', 'r') as input_file:
    lines = input_file.readlines()

# Randomly sample 500 lines
sampled_lines = random.sample(lines, 700)

# Remove sampled lines from the original list
lines = [line for line in lines if line not in sampled_lines]

# Write the sampled lines to the first output file
with open('train_data_1000_10.txt', 'w') as output_file_1:
    output_file_1.writelines(sampled_lines)

# Randomly sample 200 lines from the remaining lines
sampled_lines_2 = random.sample(lines, 200)

# Write the second set of sampled lines to the second output file
with open('validation_data_1000_10.txt', 'w') as output_file_2:
    output_file_2.writelines(sampled_lines_2)
lines2 = [line for line in lines if line not in sampled_lines and line not in sampled_lines_2]

sampled_lines_3 = random.sample(lines2, 100)
with open('test_data_1000_10.txt', 'w') as output_file_3:
    output_file_3.writelines(sampled_lines_3)


with open('mixed_2000_10.txt', 'r') as input_file:
    lines = input_file.readlines()

# Randomly sample 500 lines
sampled_lines = random.sample(lines, 1400)

# Remove sampled lines from the original list
lines = [line for line in lines if line not in sampled_lines]

# Write the sampled lines to the first output file
with open('train_data_2000_10.txt', 'w') as output_file_1:
    output_file_1.writelines(sampled_lines)

# Randomly sample 200 lines from the remaining lines
sampled_lines_2 = random.sample(lines, 400)

# Write the second set of sampled lines to the second output file
with open('validation_data_2000_10.txt', 'w') as output_file_2:
    output_file_2.writelines(sampled_lines_2)
lines2 = [line for line in lines if line not in sampled_lines and line not in sampled_lines_2]

sampled_lines_3 = random.sample(lines2, 200)
with open('test_data_2000_10.txt', 'w') as output_file_3:
    output_file_3.writelines(sampled_lines_3)

with open('mixed_3000_10.txt', 'r') as input_file:
    lines = input_file.readlines()

# Randomly sample 500 lines
sampled_lines = random.sample(lines, 2100)

# Remove sampled lines from the original list
lines = [line for line in lines if line not in sampled_lines]

# Write the sampled lines to the first output file
with open('train_data_3000_10.txt', 'w') as output_file_1:
    output_file_1.writelines(sampled_lines)

# Randomly sample 200 lines from the remaining lines
sampled_lines_2 = random.sample(lines, 600)

# Write the second set of sampled lines to the second output file
with open('validation_data_3000_10.txt', 'w') as output_file_2:
    output_file_2.writelines(sampled_lines_2)
lines2 = [line for line in lines if line not in sampled_lines and line not in sampled_lines_2]

sampled_lines_3 = random.sample(lines2, 300)
with open('test_data_3000_10.txt', 'w') as output_file_3:
    output_file_3.writelines(sampled_lines_3)
