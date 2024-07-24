import torch

# Load the data from the file
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
print("\nDepot XY:\n", depot_xy)
print("\nNode XY:\n", node_xy)
print("\nNode Demand:\n", node_demand)
print("\nNode Early TW:\n", node_earlyTW)
print("\nNode Late TW:\n", node_lateTW)
print("\nNode Service Time:\n", node_serviceTime)
print("\nRoute Open:\n", route_open)
print("\nRoute Length Limit:\n", route_length_limit)

