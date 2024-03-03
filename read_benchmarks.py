import os


def read_node_file(fopen, benchmark):
    node_info = {}
    node_info_raw_id_name ={}
    node_cnt = 0
    for line in fopen.readlines():
        if not line.startswith("\t") and not line.startswith(" "):
            continue
        line = line.strip().split()

        if "ibm" in benchmark:
            if line[0][0] == 'p':
                continue
        else:
            if line[-1] != "terminal":
                continue

        node_name = line[0]
        x = int(line[1])
        y = int(line[2])
        node_info[node_name] = {"id": node_cnt, "x": x, "y": y }
        node_info_raw_id_name[node_cnt] = node_name
        node_cnt += 1

    if "ibm" in benchmark:
        sorted_node_info = sorted(node_info.items(), key=lambda x: x[1]['x'] * x[1]['y'], reverse=True)
        node_info = dict(sorted_node_info[:256])

    print("len node_info", len(node_info))
    return node_info, node_info_raw_id_name


def read_net_file(fopen, node_info):
    net_info = {}
    net_name = None
    net_cnt = 0
    for line in fopen.readlines():
        if not line.startswith("\t") and not line.startswith("NetDegree") and not line.startswith(" "):
            continue
        line = line.strip().split()
        if line[0] == "NetDegree":
            net_name = line[-1]
        else:
            node_name = line[0]
            if node_name in node_info:
                if not net_name in net_info:
                    net_info[net_name] = {}
                    net_info[net_name]["nodes"] = {}
                    net_info[net_name]["ports"] = {}
                if not node_name in net_info[net_name]["nodes"]:
                    if len(line) == 2:
                        x_offset = 0.0
                        y_offset = 0.0
                    else:
                        x_offset = float(line[-2])
                        y_offset = float(line[-1])
                    net_info[net_name]["nodes"][node_name] = {}
                    net_info[net_name]["nodes"][node_name] = {"x_offset": x_offset, "y_offset": y_offset}
    for net_name in list(net_info.keys()):
        if len(net_info[net_name]["nodes"]) <= 1:
            net_info.pop(net_name)
    for net_name in net_info:
        net_info[net_name]['id'] = net_cnt
        net_cnt += 1
    print("adjust net size = {}".format(len(net_info)))
    return net_info


# def get_comp_hpwl_dict(node_info, net_info):
#     comp_hpwl_dict = {}
#     for net_name in net_info:
#         max_idx = 0
#         for node_name in net_info[net_name]["nodes"]:
#             max_idx = max(max_idx, node_info[node_name]["id"])
#         if not max_idx in comp_hpwl_dict:
#             comp_hpwl_dict[max_idx] = []
#         comp_hpwl_dict[max_idx].append(net_name)
#     return comp_hpwl_dict


# def get_node_to_net_dict(node_info, net_info):
#     node_to_net_dict = {}
#     for node_name in node_info:
#         node_to_net_dict[node_name] = set()
#     for net_name in net_info:
#         for node_name in net_info[net_name]["nodes"]:
#             node_to_net_dict[node_name].add(net_name)
#     return node_to_net_dict


# def get_port_to_net_dict(port_info, net_info):
#     port_to_net_dict = {}
#     for port_name in port_info:
#         port_to_net_dict[port_name] = set()
#     for net_name in net_info:
#         for port_name in net_info[net_name]["ports"]:
#             port_to_net_dict[port_name].add(net_name)
#     return port_to_net_dict


def read_pl_file(fopen, node_info):
    max_height = 0
    max_width = 0
    for line in fopen.readlines():
        if not line.startswith('o') and not line.startswith('a') and not line.startswith('p'):
            continue
        line = line.strip().split()
        node_name = line[0]
        if not node_name in node_info:
            continue
        place_x = int(float(line[1]))
        place_y = int(float(line[2]))
        max_height = max(max_height, node_info[node_name]["x"] + place_x)
        max_width = max(max_width, node_info[node_name]["y"] + place_y)
        node_info[node_name]["raw_x"] = place_x
        node_info[node_name]["raw_y"] = place_y
    return max(max_height, max_width), max(max_height, max_width)


def get_node_id_to_name_topology(node_info):
    sorted_node_info = sorted(node_info.items(), key=lambda x: x[1]['x'] * x[1]['y'], reverse=True)
    node_id_to_name_res = [node_name for node_name, _ in sorted_node_info]

    for i, node_name in enumerate(node_id_to_name_res):
        node_info[node_name]["id"] = i

    return node_id_to_name_res


def get_netlist(node_info, net_info):
    netlist = []
    for net_name, info in net_info.items():
        nodes = []
        for node_name in info["nodes"].keys():
            node_id = node_info[node_name]["id"]
            nodes.append(node_id)
        if len(nodes) > 1:
            netlist.append(nodes)
    return netlist
        
    
def generate_db_params(benchmark):
    nodesFilePath = os.path.join(".", benchmark, benchmark+".nodes")
    netsFilePath  = os.path.join(".", benchmark, benchmark+".nets")
    plFilePath    = os.path.join(".", benchmark, benchmark+".pl")
    
    nodes_fopen = open(nodesFilePath, "r")
    nets_fopen  = open(netsFilePath,  "r")
    pl_fopen    = open(plFilePath,    "r")
    
    node_info, _ = read_node_file(nodes_fopen, benchmark)
    net_info = read_net_file(nets_fopen, node_info)
    max_width, max_height = read_pl_file(pl_fopen, node_info)
    node_id_to_name = get_node_id_to_name_topology(node_info)

    netlist = get_netlist(node_info, net_info)
    return node_info, net_info, node_id_to_name, netlist, [max_width, max_height]
    