# Find facilities containing a given item

from itertools import product
import utilities

def facility_set_per_item(item_id, facilities, inventories):
    """Find facilities containing the specified item."""
    available_facilities = []
    for k in facilities:
        # If the facility has the item in stock, add it to the list of available facilities
        if inventories[k][item_id] > 0:
            available_facilities.append(k)
    return available_facilities

# Find all possible methods to fulfill (potentially partially) an order
def find_all_methods(order, facilities, inventories):
    """Find all possible methods to fulfill (potentially partially) an order."""
    all_methods = [[()]]  # Include the "do-nothing" method
    # Iterate through the items in the order
    for item in order:
        # Find facilities that have the item in stock
        facility_item = facility_set_per_item(item, facilities, inventories)
        # Add each facility-item pair as an individual method
        for k in facility_item:
            all_methods.append([(item, k)])
    # Calculate the powerset of the order to get all possible subsets
    all_subsets_order = list(utilities.powerset(order))
    for element in all_subsets_order:
        # Skip empty subsets
        if not element:
            continue
        # Get facilities that have items in the subset in stock
        facility_sets = [facility_set_per_item(i, facilities, inventories) for i in element]
        # Generate all possible facility combinations for the item subsets
        possible_facilities = list(product(*facility_sets))
        # Iterate through the facility combinations
        for facility_combo in possible_facilities:
            # If there are more than one facility in the combination
            if len(facility_combo) > 1:
                # Create a method with the item-facility pairs and append it to the list of methods
                method = [(order[n], facility_combo[n]) for n in range(len(facility_combo))]
                all_methods.append(method)
    return all_methods

# For each (order,location) find all methods. Each elements is a list of methods, with each method being a list of (i,k) tuples
def find_all_order_methods_location(df_orders_location, facility_indices, safety_stock):
    """For each (order,location) find all methods-location."""
    all_methods_location = []
    for order in df_orders_location['(order,location)']:
        order_only = order[0]  # extract only the order part from the tuple
        order_location = order[1]
        all_methods_location.append({'methods':find_all_methods(order_only, facility_indices, safety_stock), 'order': order_only, 'location': order_location, 'size': len(order_only)})
    return all_methods_location
    
    # Calculate indicator (i,k) in m
def calculate_all_indicators_i_k(all_methods_location, i_k_pair):
        """Sparse representation of indicator (i,k) in m, structured for direct indexing by q and m."""
        indicators = {}
        for q, methods_location in enumerate(all_methods_location):
            relevant_ms = set()
            for m, method in enumerate(methods_location['methods']):
                if i_k_pair in method:
                    relevant_ms.add(m)
            if relevant_ms:
                indicators[q] = relevant_ms
        return indicators

# Calculate indicators for all (i, k) pairs
def calculate_all_indicators(all_methods_location, items, facilities, safety_stock):
        """Calculate indicators for all (i, k) pairs."""
        all_indicators = {}
        for i in items:
            for k in facilities:
                if safety_stock[k][i] > 0: # Only calculate indicators for items that are in stock, else stock is zero
                    i_k_pair = (i, k)
                    all_indicators[i_k_pair] = calculate_all_indicators_i_k(all_methods_location, i_k_pair)
        return all_indicators