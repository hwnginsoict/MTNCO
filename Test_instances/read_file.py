import torch

def load_c101_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Skip the header lines
    vehicle_info_index = lines.index("VEHICLE\n")
    customer_info_index = lines.index("CUSTOMER\n")

    capacity = int(lines[vehicle_info_index + 2].split()[1])

    data_lines = lines[customer_info_index + 4:]
    depot_xy = []
    node_xy = []
    node_demand = []
    node_earlyTW = []
    node_lateTW = []
    node_serviceTime = []

    for line in data_lines:
        if line.strip():  # Skip empty lines
            parts = line.split()
            if len(depot_xy) == 0:  # First entry is the depot
                depot_xy.append([float(parts[1]), float(parts[2])])
            else:
                node_xy.append([float(parts[1]), float(parts[2])])
                node_demand.append(float(parts[3]))
                node_earlyTW.append(float(parts[4]))
                node_lateTW.append(float(parts[5]))
                node_serviceTime.append(float(parts[6]))

    # Convert to tensors
    depot_xy = torch.tensor(depot_xy).unsqueeze(0)  # Shape: (1, 1, 2)
    node_xy = torch.tensor(node_xy).unsqueeze(0)    # Shape: (1, num_nodes, 2)
    node_demand = torch.tensor(node_demand).unsqueeze(0)  # Shape: (1, num_nodes)
    node_earlyTW = torch.tensor(node_earlyTW).unsqueeze(0)  # Shape: (1, num_nodes)
    node_lateTW = torch.tensor(node_lateTW).unsqueeze(0)  # Shape: (1, num_nodes)
    node_serviceTime = torch.tensor(node_serviceTime).unsqueeze(0)  # Shape: (1, num_nodes)

    # Create route_open and route_length_limit tensors
    route_open = torch.zeros_like(node_demand)
    route_length_limit = torch.zeros_like(node_demand)

    data = {
        'depot_xy': depot_xy,
        'node_xy': node_xy,
        'node_demand': node_demand,
        'node_earlyTW': node_earlyTW,
        'node_lateTW': node_lateTW,
        'node_serviceTime': node_serviceTime,
        'route_open': route_open,
        'route_length_limit': route_length_limit
    }

    return data

# Load C101 file and save as a .pt file
c101_file_path = 'F:\\CodingEnvironment\\MTNCO\\Test_instances\\c101.txt'
data = load_c101_file(c101_file_path)
torch.save(data, 'F:\\CodingEnvironment\\MTNCO\\Test_instances\\data_VRPTW_C101.pt')

# Load the saved data to verify
loaded_data = torch.load('F:\\CodingEnvironment\\MTNCO\\Test_instances\\data_VRPTW_C101.pt')

for key, value in loaded_data.items():
    print(f"{key}: {value.shape}")

# Print the first batch to see a sample
print("\nDepot XY:\n", loaded_data['depot_xy'])
print("\nNode XY:\n", loaded_data['node_xy'])
print("\nNode Demand:\n", loaded_data['node_demand'])
print("\nNode Early TW:\n", loaded_data['node_earlyTW'])
print("\nNode Late TW:\n", loaded_data['node_lateTW'])
print("\nNode Service Time:\n", loaded_data['node_serviceTime'])
print("\nRoute Open:\n", loaded_data['route_open'])
print("\nRoute Length Limit:\n", loaded_data['route_length_limit'])
