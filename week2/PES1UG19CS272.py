"""
You can create any other helper funtions.
Do not modify the given functions
"""
from queue import PriorityQueue as pq

def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """
    # Note the total number of nodes present
    total_nodes = len(cost)
    # Handling edge cases
    if(total_nodes > 0 and len(heuristic) == total_nodes and start_point > 0 and start_point < total_nodes):
        # Dictionary to hold the path traversed upto a particular node
        path_dict = dict()
        # start_point's path will initially contain start_point itself since it is visited
        path_dict[start_point] = [start_point]
        # Data-structure used is priority queue
        priority_q = pq()
        # Bit map to help identify if a node is explored or not
        visited_nodes = [0] * total_nodes
        # f_cost = g_cost + h_cost
        # For start_point g_cost is 0
        g_cost = 0
        f_cost = g_cost + heuristic[start_point]
        # Priority queue will have a list of f_cost upto that node, g_cost_upto that node and the node number
        # Put the initial state to the priority queue
        priority_q.put([f_cost, g_cost, start_point])
        # As long as the priority queue is not empty, perform search
        while(priority_q.empty() != 1):
            # From priority queue get the minimum f_cost state
            f_cost, g_cost, min_curr_node = priority_q.get()
            # If the node is one of the goals, return the path for that goal as answer
            if(min_curr_node in goals):
                return path_dict[min_curr_node]    
            else:   
                # Otherwise obtain the adjacency list for this node
                child_nodes = cost[min_curr_node]
                # Mark the node as visited
                visited_nodes[min_curr_node] = 1
                # For all nodes in adjacency list
                for index in range(total_nodes):
                    # Check if the node is a child of min_curr_node and then if the child is not visited
                    if(child_nodes[index] > 0 and visited_nodes[index] == 0):
                        # Update the path of the child node
                        path_dict[index] = path_dict[min_curr_node] + [index]
                        # g_cost of child will be g_cost of parent + cost of parent->child
                        g_cost_child = g_cost + child_nodes[index]
                        # f_cost of child is g_cost of child + h_cost of child
                        f_cost_child = g_cost_child + heuristic[index]
                        # Put this state into the priority queue
                        priority_q.put([f_cost_child, g_cost_child, index])
    # Empty list is returned if goal state not found
    return list()

def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    # List to hold the path to one of the goal state
    path = []
    # Stack to perform DFS
    stack = []
    # Push the start_point into stack
    stack.append(start_point)
    # Note the total number of nodes present
    total_nodes = len(cost)
    # Bit map to help identify if a node is explored or not
    visited_nodes = [0] * total_nodes
    # Handling edge cases
    if(total_nodes > 0 and start_point > 0 and start_point < total_nodes):
        # Perform DFS as long as the stack is not empty
        while(len(stack) != 0):
            # Obtain the node at top of stack
            top_node = stack.pop()
            # If this node is one of the goals
            if(top_node in goals):
                # Push the node to the path list and return it as the answer
                path.append(top_node)
                return path
            else:
                # If not goal state, add it to the path list if it is not visited
                if(visited_nodes[top_node] != 1):
                    path.append(top_node)
                # Mark it as explored
                visited_nodes[top_node] = 1
                # Variable to count number of un-explored nodes from top_node
                frontier_nodes = 0
                # Variable to hold the starting index for traversal
                index = total_nodes-1
                # The nodes are traversed in a reverse manner in the adjacency list to ensure lexicographic DFS traversal
                while(index != 0):
                    # If there is an edge to top_node and the index is not visited
                    if(cost[top_node][index] > 0 and visited_nodes[index] == 0):
                        # Increment frontier_nodes
                        frontier_nodes += 1
                        # Push to stack
                        stack.append(index)
                    # Decrement the index
                    index -= 1
                
                # If no frontier_nodes found (i.e. dead-end), and the path list has nodes. Similar to back-tracking
                if(frontier_nodes == 0 and len(path) > 0):
                    # Pop nodes until a node is found having un-explored nodes
                    frontier_nodes = 0
                    while(len(path) > 0):
                        # Count un-explored nodes
                        frontier_nodes = 0
                        for i_node in range(total_nodes):
                            if(cost[path[-1]][i_node] > 0 and visited_nodes[i_node] == 0):
                                frontier_nodes += 1 
                        # If dead-end then pop from path list
                        if(frontier_nodes == 0):
                            path.pop()
                        # If un-explored nodes found then add this node to the stack to continue DFS and break
                        else:
                            stack.append(path[-1])
                            break
    # If no goal state is found or if goals list is empty return empty list
    return list()