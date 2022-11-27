import heapq
def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0
    for from_idx,value in enumerate(from_state):
        if value != 0:
            to_idx = to_state.index(value)
            from_row,from_col = int(from_idx/3),(from_idx)% 3
            to_row,to_col = int(to_idx/3),(to_idx)% 3
            distance += abs(from_row-to_row) + abs(from_col-to_col)
    return distance
def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0
    for from_idx,value in enumerate(from_state):
        if value != 0:
            to_idx = to_state.index(value)
            from_row,from_col = int(from_idx/3),(from_idx)% 3
            to_row,to_col = int(to_idx/3),(to_idx)% 3
            distance += abs(from_row-to_row) + abs(from_col-to_col)
    return distance




def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    i1 = state.index(0)
    i2 = state.index(0,i1+1)
    i1_row,i1_col = int(i1/3),(i1)% 3
    i2_row,i2_col = int(i2/3),(i2)% 3
    succ_states = []
    for i,s in enumerate(state):
        if s != 0:
            s_row,s_col = int(i/3),(i)% 3
            condition1 = (i1_row-1 == s_row or s_row == i1_row+1) and (i1_col == s_col)
            condition2 = (i1_col-1 == s_col or s_col == i1_col+1) and (i1_row == s_row)
            condition3 = (i2_row-1 == s_row or s_row == i2_row+1) and (i2_col == s_col)
            condition4 = (i2_col-1 == s_col or s_col == i2_col+1) and (i2_row == s_row)
            if condition1 or condition2:
                succ1 = state[:]
                succ1[i] = 0
                succ1[i1] = s
                succ_states.append(succ1)
            if condition3 or condition4:
                succ2 = state[:]
                succ2[i] = 0
                succ2[i2] = s
                succ_states.append(succ2)
    return sorted(succ_states)

def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    pq = []
    h = get_manhattan_distance(state)
    heapq.heappush(pq,(0+h, state, (0 , h, -1)))
    reach = False
    max_pq = []
    c = 0
    closed = dict()
    output = []
    while pq and not reach:
        max_pq.append(len(pq))
        old_cost, old_state, (old_g,old_h,old_p) = heapq.heappop(pq)
        if old_state == goal_state:
            reach = True
            output.append(f"{old_state} h={old_h} moves: {old_g}")  
            while old_p != -1:
                for (c,g,h,p),state in closed.items():
                    if c == old_p:
                        output.append(f"{state} h={h} moves: {g}")
                        old_p = p
        else:
            closed[(c,old_g,old_h,old_p)] = old_state
            g = old_g + 1 
            succ_states = get_succ(old_state)
            for succ_state in succ_states:
                if succ_state not in list(closed.values()):
                    new_h = get_manhattan_distance(succ_state)
                    heapq.heappush(pq,(g+new_h, succ_state, (g, new_h, c))) 
    
        c+=1
    output = output[::-1]
    print(*output, sep='\n')
    print(f"Max queue length: {max(max_pq)}")

