# Demand distribution for orders

import numpy as np
import random
from scipy.special import comb
import utilities

def generate_order_size_probabilities(n_max, seed_value):
    """Generate a probability distribution for the order size, including the empty order."""
    random.seed(seed_value)
    prob_size = random.sample(range(1, 1000), n_max + 1)
    sum_p = sum(prob_size)
    prob_order_size = [i / sum_p for i in prob_size]
    return prob_order_size

# Scale the demand probabilities by location
def scale_demand_probabilities_by_location(demand_distribution_by_type, population, total_population):
    demand_distribution_by_type_by_location = []
    num_order_types = len(demand_distribution_by_type)
    for n in range(num_order_types):
        temp = []
        num_orders_q = len(demand_distribution_by_type[n])
        for q in range(num_orders_q):
            # Calculate the demand probabilities for each location j and order type q
            demand = [(demand_distribution_by_type[n][q]*int(population[j]))/total_population for j in range(len(population))]
            temp.append(demand)
        # Add the demand distribution over locations for each order type q of fixed order size n
        demand_distribution_by_type_by_location.append(temp)
        
    return demand_distribution_by_type_by_location
    
# Generate oder types and a probability distribution by type for each order size n
def generate_demand_distribution(num_items, n_max, n_0, prob_order_size, seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value) # I have two sources of randomness and want to make sure they are reproducible
    order_types = [()]
    demand_distribution_by_type = [[prob_order_size[0]]] # order of size 0 equally likely across types
    
    for size in range(1, n_max + 1):
        max_combinations = int(comb(num_items, size))
        target_combinations = min(n_0, max_combinations)
        unique_orders = set()
        
        while len(unique_orders) < target_combinations:
            order = utilities.random_combination(range(num_items), size)
            unique_orders.add(order)
        
        # Create a list of n_0 probabilities for each order size
        random_prob_q = np.random.uniform(0.00001, prob_order_size[size], size=target_combinations)
        temp = []
        for index, order in enumerate(unique_orders):
            if order not in order_types:
                order_types.append(order)
                prob_order_q = (random_prob_q[index]*prob_order_size[size]) / random_prob_q.sum()
                temp.append(prob_order_q)
        demand_distribution_by_type.append(temp)
        
    return order_types, demand_distribution_by_type

def adjust_arrival_probabilities(T, alpha, demand_distribution_by_type_by_location):
    """
    Adjust the arrival probabilities based on whether we are in the first or second half of fulfillment and the order size.
    Normalizes the probabilities across all order types and locations for each time period t.
    """
    # Initialize the structure to hold the adjusted probabilities
    adjusted_demand_distribution_by_type_by_location = []

    # Determine scaling factors based on the time period
    alpha_scale_first_half = 2 * alpha
    alpha_scale_second_half = 2 * (1 - alpha)

    # Loop over each time period
    for t in range(1, T + 1):
        temp_probs_by_time_period = []  # Temporary list for the adjusted probabilities in the current period by order size

        # Store all probabilities for normalization purposes
        all_probs_for_normalization = []

        # Loop over each order size
        for n, order_types_probs in enumerate(demand_distribution_by_type_by_location):
            temp_probs_by_order_type = []  # Temp list for the current order size

            # Loop over each order type for the current order size
            for q, probs_by_location in enumerate(order_types_probs):
                adjusted_probs_by_location = []  # Adjust the probabilities based on the order size and the time period
                
                for j, prob in enumerate(probs_by_location):
                    if n == 0:
                        # Leave the zero order size probabilities as they are
                        adjusted_prob = prob
                    elif n == 1:
                        # Scale probabilities for order size of 1
                        adjusted_prob = alpha_scale_first_half * prob if 1 <= t <= T // 2 else alpha_scale_second_half * prob
                    else:
                        # Scale probabilities for order size greater than 1
                        adjusted_prob = alpha_scale_second_half * prob if 1 <= t <= T // 2 else alpha_scale_first_half * prob
                    adjusted_probs_by_location.append(adjusted_prob)

                temp_probs_by_order_type.extend(adjusted_probs_by_location)

            # Collect all probabilities for normalization
            all_probs_for_normalization.extend(temp_probs_by_order_type)

        # Normalize the collected probabilities so they sum to 1
        total_prob = sum(all_probs_for_normalization)
        normalized_probs = [prob / total_prob for prob in all_probs_for_normalization] if total_prob > 0 else all_probs_for_normalization

        # Re-distribute the normalized probabilities back to the original structure
        index = 0
        for n, order_types_probs in enumerate(demand_distribution_by_type_by_location):
            temp_probs_by_order_type = []
            for q, probs_by_location in enumerate(order_types_probs):
                normalized_probs_by_location = normalized_probs[index:index+len(probs_by_location)]
                index += len(probs_by_location)
                temp_probs_by_order_type.append(normalized_probs_by_location)
            temp_probs_by_time_period.append(temp_probs_by_order_type)

        # Add the adjusted and normalized probabilities for the current period to the overall structure
        adjusted_demand_distribution_by_type_by_location.append(temp_probs_by_time_period)

    return adjusted_demand_distribution_by_type_by_location

def aggregate_demand_distribution(adjusted_demand_distribution_by_type_by_location): 
    # Initialize the structure for the new distribution
    aggregate_demand_distribution = []

    # Iterate over the order sizes
    for n in range(len(adjusted_demand_distribution_by_type_by_location[0])):
        new_order_type_probs = []

        # Iterate over the order types for the current order size
        for q in range(len(adjusted_demand_distribution_by_type_by_location[0][n])):
            new_probs_by_location = [0] * len(adjusted_demand_distribution_by_type_by_location[0][n][q])

            # Iterate over each period and accumulate probabilities
            for period in adjusted_demand_distribution_by_type_by_location:
                for loc_idx, prob in enumerate(period[n][q]):
                    new_probs_by_location[loc_idx] += prob

            new_order_type_probs.append(new_probs_by_location)
        
        aggregate_demand_distribution.append(new_order_type_probs)

    return aggregate_demand_distribution


