import torch
import os

def load_file(file_path, num_nodes):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Skip the header lines
    vehicle_info_index = lines.index("VEHICLE\n")
    customer_info_index = lines.index("CUSTOMER\n")

    capacity = int(lines[vehicle_info_index + 2].split()[1])

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

            node_demand.append(float(parts[3])/capacity)
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
    num_node = num_nodes
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


output_dir = 'F:\\CodingEnvironment\\MTNCO\\Test_instances\\mock_data\\'
os.makedirs(output_dir, exist_ok=True)

file_path = 'F:\\CodingEnvironment\\MTNCO\Baseline\\VRPTW\POMO\\Mock_data\\'

for num in [20, 50, 100, 150, 200]:
    for i in range(1,4):
        data = load_file(f"{file_path}{num}\\{str(i)}.txt", num)
        torch.save(data, f"{output_dir}data_VRPTW_{num}_{i}.pt")

