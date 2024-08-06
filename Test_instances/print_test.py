import torch

# Load the data from the file
# file_path = 'F:\CodingEnvironment\MTNCO\Test_instances\Solomon100\data_VRPTW_c101.pt'
file_path = 'F:\CodingEnvironment\MTNCO\Test_instances\data_VRPTW_100_5000.pt'
data = torch.load(file_path)

# Print the data to understand its structure
# Print the data to understand its structure
for key, value in data.items():
    print(f"{key}: {value.shape}")

# Example of accessing specific elements
depot_xy = data['depot_xy']
node_xy = data['node_xy']
node_demand = data['node_demand']
node_earlyTW = data['node_earlyTW']
node_lateTW = data['node_lateTW']
node_serviceTime = data['node_serviceTime']
route_open = data['route_open']
route_length_limit = data['route_length_limit']

# Print the first batch to see a sample
print("\nDepot XY:\n", depot_xy[0])
print("\nNode XY:\n", node_xy[0])
print("\nNode Demand:\n", node_demand[0])
print("\nNode Early TW:\n", node_earlyTW[0])
print("\nNode Late TW:\n", node_lateTW[0])
print("\nNode Service Time:\n", node_serviceTime[0])
print("\nRoute Open:\n", route_open[0])
print("\nRoute Length Limit:\n", route_length_limit[0])

