import os

def process_directory(root_directory, local_embedding):
    for root, _, files in os.walk(root_directory):
        for file in files:
            print(file)


root_directory = "all_data"
local_embedding = True

process_directory(root_directory, local_embedding)