import numpy as np

# divide turbines into groups according to geological locations
hierarchy_dict1 = {
    'c1': list(range(113, 135)),
    'c2': list(range(91, 113)),
    'c3': list(range(69, 91)),
    'c4': list(range(1, 25)),
    'c5': list(range(25, 48)),
    'c6': list(range(48, 68)),
    'total': list(range(1,135))
}

hierarchy_dict2 = {
    'c1': list(range(113, 135)),
    'c2': list(range(91, 113)),
    'c3': list(range(69, 91)),
    'c4': list(range(1, 25)),
    'c41': [9,10,11,12],
    'c5': list(range(25, 48)),
    'c51': list(range(31,36)),
    'c6': list(range(48, 68)),
    'c61': list(range(52,58)),
    'total': list(range(1,135))
}


hierarchy_dict3 = {
    'd1': list(range(113, 124)),
    'd2': list(range(128,135)) + list(range(113,116)),
    'd3': list(range(124,135)),
    'd4': list(range(113,124)) + list(range(91, 102)),
    'd5': list(range(124, 135)) + list(range(102, 113)),
    'd6': list(range(102, 113)) + list(range(80,91)),
    'd7': list(range(91,102)) + list(range(69,80)),
    'd8': list(range(69,80)) + list(range(1,13)),
    'd9': list(range(80,91)) + list(range(13, 24)),
    'd10': list(range(1,13)) + list(range(25,36)),
    'd11': list(range(13, 24)) + list(range(36,48)),
    'd12':  list(range(25,36)) + list(range(48, 58)),
    'd13': list(range(36,48)) + list(range(58, 69)),
    'd14':  [9,10,11,12] + list(range(31,36)) + list(range(52,58)),
    'c1': list(range(113, 135)),
    'c2': list(range(91, 113)),
    'c3': list(range(69, 91)),
    'c4': list(range(1, 25)),
    'c41': [9,10,11,12],
    'c5': list(range(25, 48)),
    'c51': list(range(31,36)),
    'c6': list(range(48, 68)),
    'c61': list(range(52,58)),
    'total': list(range(1,135))
}

def hierarchy_dict_to_matrix(base_nodes, dic):
    """
    base_nodes: basic nodes
    dic: relationship of upper nodes to base nodes
    return: 
    Su: y_upper = Suy_b
    A: Ay=0
    upper_nodes: all nodes that are not base nodes
    all_nodes: all nodes incluiding base nodes
    """
    bc = len(base_nodes) 
    base_nodes_index = dict(zip(base_nodes, range(len(base_nodes))))  
    upper_nodes = sorted(list(dic.keys()), key=lambda key:len(dic[key]))
    all_nodes = base_nodes + upper_nodes
    
    uc = len(upper_nodes)
    Su = np.zeros([uc, bc])
    
    for i in range(len(upper_nodes)):
        u_node = upper_nodes[i]
        u_base_nodes = dic[u_node]
        for base_node in u_base_nodes:
            j = base_nodes_index[base_node]
            Su[i][j] = 1
    
    A = np.hstack([-Su, np.eye(uc)])
    return Su, A, upper_nodes, all_nodes
