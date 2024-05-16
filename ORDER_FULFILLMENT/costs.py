# Calculate costs and distances

import numpy as np
import utilities

def calculate_costs_and_distances(city_locations, facility_locations):
    """Calculate fixed costs, variable costs, and distances between cities and fulfillment centers."""
    num_cities = len(city_locations)
    num_facilities = len(facility_locations)
    c_fixed = np.zeros((num_facilities + 1, num_cities))  # Matrix of fixed costs (including the dummy facility at the end)
    c_unit = np.zeros((num_facilities + 1, num_cities))  # Matrix of variable costs (including the dummy facility at the end)
    dists = np.zeros((num_facilities, num_cities))  # Matrix of distances between cities and facilities
    for j in range(num_cities):
        dist_max = 0
        for k in range(num_facilities):
            c_fixed[k][j] = 8.759
            dists[k][j] = utilities.haversine(facility_locations[k][0], facility_locations[k][1],
                                    city_locations[j][0], city_locations[j][1], inv=1)
            c_unit[k][j] = 0.423 + 0.000541 * dists[k][j]
            dist_max = max(dist_max, dists[k][j])
        # Routing to FC K+1 corresponds to not fulfilling an item
        c_fixed[num_facilities][j] = 2 * 8.759  # Twice the fixed cost
        c_unit[num_facilities][j] = 2 * (0.423 + 0.000541 * dist_max)  # Twice the variable cost of fulfilling from the farthest facility
    return dists, c_fixed, c_unit

# Calculate costs for all methods
def calculate_all_costs(num_cities, all_methods_location, c_fixed, c_unit):
    """
    Calculate costs for all methods.
    """
    all_costs = [[0]]*num_cities # cost of empty order (from anywhere) is zero
    for methods_location in all_methods_location[num_cities:]: # exclude empty orders because we already added the cost of empty orders
        cost_methods = []
        for method in methods_location['methods']:
            if method == [()]:
                # cost of not fulfilling an order
                cost_methods.append((c_fixed[-1][methods_location['location']] +
                                    c_unit[-1][methods_location['location']])*methods_location['size']) # highest possible cost
            else:
                variable_cost = 0
                fixed_cost = 0
                facilities_used = set()
                for item, facility in method:
                    variable_cost += c_unit[facility][methods_location['location']]
                    facilities_used.add(facility) # add facility to set of facilities used (only if not there yet)
                fixed_cost = sum(c_fixed[facility][methods_location['location']] for facility in facilities_used)
                cost = variable_cost + fixed_cost
                if len(method) != methods_location['size']:  # calculate cost if partially fulfilled
                    cost += c_fixed[-1][methods_location['location']] # Add fixed cost for not fulfilling the remaining items once
                    cost += c_unit[-1][methods_location['location']] * (methods_location['size'] - len(method)) # Add variable cost for each unfulfilled item
                cost_methods.append(cost)
        all_costs.append(cost_methods)
    return all_costs