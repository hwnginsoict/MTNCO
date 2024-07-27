import torch

def load_c101_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Skip the header lines
    vehicle_info_index = lines.index("VEHICLE\n")
    customer_info_index = lines.index("CUSTOMER\n")

    capacity = int(lines[vehicle_info_index + 2].split()[1])/260

    data_lines = lines[customer_info_index + 3:]
    depot_xy = []
    node_xy = []
    node_demand = []
    node_earlyTW = []
    node_lateTW = []
    node_serviceTime = []
    for line in data_lines:
        print(line)

    for line in data_lines:
        if line.strip():  # Skip empty lines
            parts = line.split()
            if len(depot_xy) == 0:
                depot_xy.append([float(parts[1])/100, float(parts[2])/100])
            else: 
                node_xy.append([float(parts[1])/100, float(parts[2])/100])

            node_demand.append(float(parts[3])/260)
            node_earlyTW.append(float(parts[4])/100)
            node_lateTW.append(float(parts[5])/100)
            node_serviceTime.append(float(parts[6])/100)

    # Convert to tensors
    depot_xy = torch.tensor(depot_xy).unsqueeze(0).clone().detach()  # Only the first line is depot
    node_xy = torch.tensor(node_xy).unsqueeze(0).clone().detach()   # The rest are nodes
    node_demand = torch.tensor(node_demand[1:]).unsqueeze(0).clone().detach()
    node_earlyTW = torch.tensor(node_earlyTW[1:]).unsqueeze(0).clone().detach()
    node_lateTW = torch.tensor(node_lateTW[1:]).unsqueeze(0).clone().detach()
    node_serviceTime = torch.tensor(node_serviceTime[1:]).unsqueeze(0).clone().detach()

    # Expand the tensors to match the desired size
    batch_size = 5000
    num_node = 100
    depot_xy = depot_xy.expand(batch_size, 1, 2)
    node_xy = node_xy.expand(batch_size, num_node, 2)
    node_demand = node_demand.expand(batch_size, num_node)
    node_earlyTW = node_earlyTW.expand(batch_size, num_node)
    node_lateTW = node_lateTW.expand(batch_size, num_node)
    node_serviceTime = node_serviceTime.expand(batch_size, num_node)

    # Create route_open and route_length_limit tensors
    route_open = torch.zeros_like(node_demand)
    route_length_limit = torch.zeros(batch_size, node_demand.size(1) + 1)

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
c101_file_path = 'F:\CodingEnvironment\MTNCO\Baseline\VRPTW\POMO\C100\c101.txt'
data = load_c101_file(c101_file_path)
torch.save(data, 'F:\CodingEnvironment\MTNCO\Test_instances\data_VRPTW_C101.pt')

# Load the saved data to verify
loaded_data = torch.load('F:\CodingEnvironment\MTNCO\Test_instances\data_VRPTW_C101.pt')

for key, value in loaded_data.items():
    print(f"{key}: {value.shape}")

# Print the first batch to see a sample
print("\nDepot XY:\n", loaded_data['depot_xy'][0])
print("\nNode XY:\n", loaded_data['node_xy'][0])
print("\nNode Demand:\n", loaded_data['node_demand'][0])
print("\nNode Early TW:\n", loaded_data['node_earlyTW'][0])
print("\nNode Late TW:\n", loaded_data['node_lateTW'][0])
print("\nNode Service Time:\n", loaded_data['node_serviceTime'][0])
print("\nRoute Open:\n", loaded_data['route_open'][0])
print("\nRoute Length Limit:\n", loaded_data['route_length_limit'][0])
