import os
import re
from collections import defaultdict

# Define the directory containing the files
directory = "test/stanford_format/"

# Option to delete incorrect format files
delete_incorrect_files = False

# ------------------------------------------------------------------------------
# Create a dictionary to hold the files, organized by timestamp
files_dict = defaultdict(lambda: defaultdict(lambda: [False]*6))

# Define expected types
expected_types = ['I', 'Q', 'U', 'V']

# Counter for incorrect format files
incorrect_format_count = 0

# Flag to track if any issues are found
issues_found = False

# Loop through each file in the directory
for filename in os.listdir(directory):
    match = re.match(r"^hmi\.s_720s\.(\d{4}\.\d{2}\.\d{2}_\d{2}:\d{2}:\d{2}_TAI)\.([IQUV])(\d)\.fits$", filename)
    if match:
        timestamp, file_type, index = match.groups()
        files_dict[timestamp][file_type][int(index)] = True
    else:
        print(f"Incorrect format: {filename}")
        incorrect_format_count += 1
        issues_found = True
        if delete_incorrect_files:
            os.remove(os.path.join(directory, filename))
            print(f"Deleted: {filename}")

# Print the total number of incorrect format files
print(f"Total number of incorrect format files: {incorrect_format_count}")

# Check for missing files
for timestamp, types in files_dict.items():
    for expected_type in expected_types:
        for index in range(6):
            if not types[expected_type][index]:
                print(f"Missing {expected_type}{index} for timestamp {timestamp}")
                issues_found = True

# Output message if no issues are found
if not issues_found:
    print("Everything is good")