import numpy as np

def dist(points, p1, p2=None, ):
    """ Compute the distance between a point and a list of points, or two points, depending
        on the value of p2. The reason for this choice is that if we want to compute the
        distance between p1 and a subset of points, we cannot extract the coordinates of
        p1 from the whole list of points, since we only access the subset (for efficiency).
        Hence, we provide the coordinates directly, instead of the city index

    Args:
        points (array): 2D array of points (each line corresponds to a point)
        p1 (int | array): a point or an index, depending on the value of p2
        p2 (int, optional): Second point, in case we want to compute distance between two points. Defaults to None.

    Returns:
        float: squared distance
    """
    if p2 is None:
        # In case we give 1 point, it must be the actual coordinates
        return np.sum((points - p1) ** 2, axis=1)
    else:
        # When 2 points are given, they are the indices
        return np.sum((points[p1] - points[p2]) ** 2)
# ----------------------------------------------------------------------------------------------------------------------------

def cost(diameter, lambda_, n):
    """Computes the cost term of the function  to optimize

    Args:
        diameter (float): Squared distance between the two points on diameter (also called support points)
        lambda_ (float): weighting coefficient of the cost, fixed value before simulation
        n (int): Total number of cities

    Returns:
        float: Cost of the deployement
    """
    return lambda_ * n * np.pi * diameter / 4
# ----------------------------------------------------------------------------------------------------------------------------


def cost_diff(sel_city, state, supp_1, supp_2, points, rewards, lambda_, city_range):
    """ Compute the cost difference between the current state, and the proposed state
        where we add (if not currently selected) or remove (if selected) a state. Also returns the two
        points with maximal distance. This is the most complicated part of the algorithm. A more
        detailed explanation can be found in the report, in the last section.

    Args:
        sel_city (int): Selected city for the proposed move
        state (array): Current state, represented as a 0-1 N-dimensional array, where N is the number of cities
        supp_1 (int): Index of the first support point (points with maximal distance)
        supp_2 (int): Index of the second support point (points with maximal distance)
        points (array): 2D array containing the coordinates of each poin in the euclidean plane
        rewards (array): Reward associated with each city, that is, expected return on investiment if we include cities in deploy. area
        lambda_ (float): Weight of the cost term
        city_range (array): Range containing all indices of cities. As it is required multiple times during the execution, it is better to always use the same.

    Returns:
        (float, int, int): (Change in the cost if we decided to accept the proposed, support point 1 if move accepted, supp. point 2 if move acc.) 
    """
    # Check if we want to remove or add the city
    want_add = state[sel_city] == 0
    
    if want_add:
        # if we have less than 2 points -> one of the supports has to be None
        if supp_1 is None and supp_2 is None:
            return rewards[sel_city], sel_city, None
        elif supp_2 is None:
            return rewards[sel_city] - cost(dist(points, supp_1, sel_city), lambda_, len(rewards)), supp_1, sel_city
        # We make sure that if one of the support is None, it is the second one
        
        # Otherwise compare distance with current points. 
        # If new max distance is larger, new point is
        # a support -> just find the other one
        
        
        state[sel_city] = 1
        dist_to_new = dist(points[state == 1, :], points[sel_city])
        curr_cities = city_range[state == 1]
        max_dist = np.max(dist_to_new)
        diameter = dist(points, supp_1, supp_2)
        
        # restore state
        state[sel_city] = 0
        if max_dist > diameter:
            # Need to change the support
            nsupp_1 = sel_city
            nsupp_2 = np.argmax(dist_to_new)

            # Warning: nsupp_2 is currently the index in the list of the points currently selected but we want index in the whole list!
            
            #print(nsupp_2)
            #print(curr_cities)
            nsupp_2 = curr_cities[nsupp_2]
            n_diameter = dist(points, nsupp_1, nsupp_2)
            return rewards[sel_city] - cost(n_diameter, lambda_, len(city_range)) + cost(diameter, lambda_, len(city_range)), nsupp_1, nsupp_2
        else:
            # Keep same support, only has reward
            return rewards[sel_city], supp_1, supp_2
    else:
        # if we don't remove a support point, then we only decrease score and keep support
        if sel_city not in (supp_1, supp_2):
            return - rewards[sel_city], supp_1, supp_2
        else:
            #print("rm supp")
            # If we have too few points, we might remove the support points -> care with that
            if sum(state) == 2:
                # Keep one supp point only -> no more cost, only the reward of the remaining state
                removed_supp = supp_1 if supp_1 == sel_city else supp_2
                remain_supp = supp_2 if supp_1 == sel_city else supp_1
                return - rewards[removed_supp] + cost(dist(points, supp_1, supp_2), lambda_, len(rewards)), remain_supp, None
            elif sum(state) == 1:
                # We remove a support point and we only have one support point -> no support left
                return - rewards[supp_1], None, None
            # Otherwise, we have more that 2 states remaining
            # To find the maximal distance between points in remaining points, use "2-hops distance" (see report why)
            state[sel_city] = 0
            remaining_supp = supp_1 if sel_city == supp_2 else supp_2
            sub_points = points[state == 1]
            intermediate_dist = dist(sub_points, points[remaining_supp])
            intermediate_supp = np.argmax(intermediate_dist)
            sel_cities = city_range[state == 1]
            intermediate_supp = sel_cities[intermediate_supp]
            # compute 2-hops distance -> it is the actual max
            dist_to_interm = dist(sub_points, points[intermediate_supp])
            other_supp = np.argmax(dist_to_interm)
            other_supp = sel_cities[other_supp]
            state[sel_city] = 1
            
            nsupp_1 = intermediate_supp
            nsupp_2 = other_supp
            
            old_cost = cost(dist(points, supp_1, supp_2), lambda_, len(city_range))
            new_cost = cost(dist(points, nsupp_1, nsupp_2), lambda_, len(city_range))
            
            return -rewards[sel_city] - new_cost + old_cost, nsupp_1, nsupp_2
# ----------------------------------------------------------------------------------------------------------------------------



def flip(x):
    """Inverts the value of the state. If x == 0 -> 1, if x == 1 -> 0

    Args:
        x (int): Binary value, representing if we consider a city in our deployement area

    Returns:
        int: 0 or 1 depending on the value of x
    """
    return 1 - x # = 0 when x == 1 and 1 when x == 0
# ----------------------------------------------------------------------------------------------------------------------------

def is_move_accepted(sel_city, state, supp_1, supp_2, points, rewards, lambda_, city_range, unif_rv, beta):
    """ Check wheter a move is accepted or not using the cost_diff function, the current beta parameter
        of the simulated annealing and a uniform random variable for the randomness of the decision

    Args:
        sel_city (int): Selected city for the proposed move
        state (array): Current state, represented as a 0-1 N-dimensional array, where N is the number of cities
        supp_1 (int): Index of the first support point (points with maximal distance)
        supp_2 (int): Index of the second support point (points with maximal distance)
        points (array): 2D array containing the coordinates of each poin in the euclidean plane
        rewards (array): Reward associated with each city, that is, expected return on investiment if we include cities in deploy. area
        lambda_ (float): Weight of the cost term
        city_range (array): Range containing all indices of cities. As it is required multiple times during the execution, it is better to always use the same.
        unif_rv (float): Uniformly distributed RV on [0, 1]
        beta (float): Simulated annealing parameter. Controls how biased in favor of best states the distribution is. Different values -> different mixing times

    Returns:
        (bool, int, int): (whether or not the move is accepted, new support point 1, new support point 2)
    """
    delta_f, nsupp_1, nsupp_2 = cost_diff(sel_city, state, supp_1, supp_2, points, rewards, lambda_, city_range)

    
    # Note: we have beta here instead of - beta because we maximize the function, as opposed to the Ising model where we minimize it.
    #accept_prob: sigmoid of the cost change
    #accept_prob = 1/2 * ( 1 + np.tanh(beta * delta_f))
    accept_prob = np.exp(beta * delta_f)
    return unif_rv <= accept_prob, nsupp_1, nsupp_2
# ----------------------------------------------------------------------------------------------------------------------------

def rw_step(state, supp_1, supp_2, points, rewards, lambda_, city_range, beta):
    """Executes one step of the random walk

    Args:
        state (array): Current state, represented as a 0-1 N-dimensional array, where N is the number of cities
        supp_1 (int): Index of the first support point (points with maximal distance)
        supp_2 (int): Index of the second support point (points with maximal distance)
        points (array): 2D array containing the coordinates of each poin in the euclidean plane
        rewards (array): Reward associated with each city, that is, expected return on investiment if we include cities in deploy. area
        lambda_ (float): Weight of the cost term
        city_range (array): Range containing all indices of cities. As it is required multiple times during the execution, it is better to always use the same.
        beta (float): Simulated annealing parameter. Controls how biased in favor of best states the distribution is. Different values -> different mixing times

    Returns:
        (int, int): (new support point 1, new support point 2)
    """
    # Generate random quantities required to work
    sel_city = np.random.choice(city_range)
    unif_rv = np.random.uniform()
    # Check if proposed move is accepted, and what would the new support points be
    is_ma, nsupp_1, nsupp_2 = is_move_accepted(sel_city, state, supp_1, supp_2, points, rewards, lambda_, city_range, unif_rv, beta)
    # if move accepted, modify the state
    if is_ma:
        state[sel_city] = flip(state[sel_city])
        return nsupp_1, nsupp_2
    else:
        # If rejected, just return the current support
        return supp_1, supp_2
# ----------------------------------------------------------------------------------------------------------------------------

def all_pairs_sq_dist(points):
    """ Compute the squared distance between all pairs of points. Use with few points only. Also compute between which points
    we have this distance

    Args:
        points (array): 2D array containing all the points. Each row represent a point

    Returns:
    float, int, int: (max distance between points, max point 1, max point 2)
    """
    n_points = len(points)
    p1 = points[np.newaxis, :, :]
    p2 = points[:, np.newaxis, :]
    res = np.sum((p1 - p2) ** 2, axis=2)
    max_index = np.argmax(res)
    row = max_index // n_points
    col = max_index % n_points
    return res, row, col
# ----------------------------------------------------------------------------------------------------------------------------


def optimize(points, rewards, lambda_, burning_steps, betas, initial_nb_states=5, seed=999):
    """Compute the full simulated annealing and return the optimal solution found

    Args:
        points (array): 2D array containing all the points. Each row represents a point
        rewards (array): Rewards associated with each city
        lambda_ (float): Weigthing coefficient of the cost of a deployement
        burning_steps (array): List of number of "burned" steps of the simulated annealing for each diff. beta before being close enough to stat. distr
        betas (array): List of successiv values of the beta used during the simulated annealing
        initial_nb_states (int, optional): Number of states in the initial position. Defaults to 5.
        seed (int, optional): Seed of the random generator. Used for reproducibility. Defaults to 999.

    Returns:
        (array, int, int): (best state found by simulated annealing, support point 1, support point 2)
    """
    np.random.seed(seed)
    N = points.shape[0]
    city_range = np.arange(N)
    initial_cities = np.random.permutation(city_range)[:initial_nb_states]
    # Create state representation & signal that cities are selected
    state = np.zeros(N)
    state[initial_cities] = 1
    selected_points = points[initial_cities, :]

    supp_1, supp_2 = None, None
    if initial_nb_states >= 2:
        _, supp_1, supp_2 = all_pairs_sq_dist(selected_points)
        # supp_1 and supp_2 represent the maxs in the subset, we need to know what there indice in the whole is
        supp_1 = initial_cities[supp_1]
        supp_2 = initial_cities[supp_2]
    elif initial_nb_states == 1:
        supp_1 = initial_cities[0]
        supp_2 = None
    else:
        supp_1 = None
        supp_2 = None
    

    # Run the simulation a certain number of steps for each beta value and return the final state
    for beta, b in zip(betas, burning_steps):
        print(f"Starting simulated annealing step with beta={beta}")
        for _ in range(b):
            supp_1, supp_2 = rw_step(state, supp_1, supp_2, points, rewards, lambda_, city_range, beta)

            
    return state, supp_1, supp_2
# ----------------------------------------------------------------------------------------------------------------------------
        