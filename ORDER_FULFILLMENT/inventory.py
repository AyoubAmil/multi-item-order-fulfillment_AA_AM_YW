# Generate inventory

import numpy as np
import utilities

# INVENTORY DATA for each facility k and item i
def generate_inventory_data(facility_indices, items, p_stock, seed_value):
    """Generate inventory data for each facility and item."""
    data_inventory = []
    np.random.seed(seed_value)
    for k in facility_indices:
        for i in items:
            # Determine if the item is potentially in stock at facility k with probability p_stock
            inv = np.random.binomial(1, p=p_stock)
            data_inventory.append({"facility": k, "product_id": i, "potentially-in-stock": inv})
    return data_inventory

# Calculate the set of cities served by each facility for each item
def get_location_customers_by_facility_and_item(facility_indices, items, city_locations, facility_locations, df_inventory, num_items):
    """
    Gets the set of cities served by each facility for each item.
    """
    location_customers_by_facility_and_item = []
    for k in facility_indices:
        temp_k = []
        for i in items:
            temp_i = []
            for j in range(len(city_locations)):
                # Calculate the distance from the closest facility to city j
                distance_closest_facility_to_j = round(min(utilities.haversine(facility_locations[k_1][0], facility_locations[k_1][1], 
                                                                    city_locations[j][0], city_locations[j][1], 
                                                                    df_inventory["potentially-in-stock"][k_1*num_items+i]) for k_1 in facility_indices),4)
                # Calculate the distance from facility k to city j
                distance_k_j = round(utilities.haversine(facility_locations[k][0], facility_locations[k][1], 
                                            city_locations[j][0], city_locations[j][1], df_inventory["potentially-in-stock"][k*num_items+i]),4)
                if distance_k_j == distance_closest_facility_to_j:
                    temp_i.append(j)
            temp_k.append(temp_i)
        location_customers_by_facility_and_item.append(temp_k)
    return location_customers_by_facility_and_item

def calculate_expected_demand(n_0, facility_indices, items, location_customers_by_facility_and_item, order_types, adjusted_demand_distribution_by_type_by_location):
        """
        Calculate the total incoming expected demand for each facility and item.
        """
        expected_demand_k_i = []
        for k in facility_indices:
            temp_k = []
            for i in items:
                prob = 0
                # Iterate over cities served by the facility for the current item
                for j in location_customers_by_facility_and_item[k][i]:
                    # Iterate over demand types
                    for order in order_types:
                        if i in order:
                            index_order = order_types.index(order)
                            # Calculate average probability of the order occurring at city j over all time periods
                            avg_prob = sum(adjusted_demand_distribution_by_type_by_location[t][len(order)][index_order-n_0*(len(order)-1)-1][j] for t in range(len(adjusted_demand_distribution_by_type_by_location))) / len(adjusted_demand_distribution_by_type_by_location)
                            prob += avg_prob
                temp_k.append(prob)
            expected_demand_k_i.append(temp_k)
        return expected_demand_k_i
    
def calculate_safety_stock(facility_indices, items, T, expected_demand_k_i):
        """
        Calculate inventory level.
        """
        S = []
        for k_idx, k in enumerate(facility_indices):
            temp_k = []
            for i_idx, i in enumerate(items):
                
                mu = T * expected_demand_k_i[k_idx][i_idx]
                mu = np.ceil(mu)
                
                temp_k.append(mu)
            S.append(temp_k)
        return S