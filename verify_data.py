import os
import re
from collections import defaultdict

# Define the directory containing the files
directory = "./Dataset"

# Create a dictionary to hold the files, organized by timestamp
files_dict = defaultdict(lambda: defaultdict(lambda: [False]*6))

# Define expected types
expected_types = ['I', 'Q', 'U', 'V']

# Loop through each file in the directory
for filename in os.listdir(directory):
    match = re.match(r"hmi\.S_720s\.(\d{8}_\d{6}_TAI)\.1\.([IQUV])(\d)\.fits", filename)
    if match:
        timestamp, file_type, index = match.groups()
        files_dict[timestamp][file_type][int(index)] = True

# Check for missing files
for timestamp, types in files_dict.items():
    for expected_type in expected_types:
        for index in range(6):
            if not types[expected_type][index]:
                print(f"Missing {expected_type}{index} for timestamp {timestamp}")
