# order_fulfillment_environment.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product, chain, combinations
import scipy.stats as sps
import random
from math import *
from scipy.special import comb

class OrderFulfillment:
    def __init__(self, num_items=20, n_max=5, n_0=5, p_stock=0.75, T=10**4, CSL=0.5, facilities_data="Data/fulfillment_centers_warmup.csv", cities_data="Data/cities_warmup.csv", prob_seed_value=1, order_seed_value=1, inv_seed_value=1, alpha=0.5):
        self.prob_seed_value = prob_seed_value # Used for reproducibility of arrival probabilities
        self.order_seed_value = order_seed_value # Used for reproducibility of order types
        self.inv_seed_value = inv_seed_value # Used for reproducibility of inventories
        self.num_items = num_items # Number of items
        self.n_max = n_max # Maximum order size
        self.n_0 = n_0 # Number of order types for each order size
        self.p_stock = p_stock # Probability that an item is in stock at a facility
        self.T = T # Time horizon
        self.CSL = CSL # Cycle Service level (used for safety stock formula)
        self.facilities_data = facilities_data # Path to the CSV file containing the facility data
        self.cities_data = cities_data # Path to the CSV file containing the city data
        self.alpha = alpha # Parameter for the transformation of the arrival probabilities
        self.items = self.generate_items_list(self.num_items) # List of item indices
        self.fulfillment_centers = self.load_fulfillment_centers(self.facilities_data, use_single_facility=False) # Dataframe of fulfillment centers
        self.facility_indices, self.facility_locations = self.generate_facility_indices_and_locations(self.fulfillment_centers) # List of fulfillment center indices and locations
        self.num_facilities = len(self.facility_indices) # Number of fulfillment centers
        self.cities = self.load_cities(self.cities_data, use_single_city=False) # Dataframe of cities
        self.city_indices, self.city_locations = self.generate_city_indices_and_locations(self.cities) # List of city indices and locations
        self.num_cities = len(self.city_indices) # Number of cities
        self.population, self.total_population = self.get_population_data(self.cities) # List of city populations and total population
        self.distances, self.fixed_costs, self.unit_costs = self.calculate_costs_and_distances(self.city_locations, self.facility_locations) # Costs and distances between cities and fulfillment centers
        self.prob_order_size = self.generate_order_size_probabilities(self.n_max, self.prob_seed_value) # Probability distribution for the order size, including the empty order
        self.order_types, self.demand_distribution_by_type = self.generate_demand_distribution(self.n_max, self.n_0, self.prob_order_size, self.order_seed_value) # Probability distribution by type for each order size n
        self.demand_distribution_by_type_by_location = self.scale_demand_probabilities_by_location(self.demand_distribution_by_type, self.population, self.total_population) # Demand distribution by type by location
        self.adjusted_demand_distribution_by_type_by_location = self.adjust_arrival_probabilities(self.T) # Adjust the arrival probabilities based on whether we are in the first or second half of fulfillment and the order size
        self.df_orders = self.create_orders_df(self.order_types) # Dataframe of order types
        self.df_orders_location = self.create_orders_location_df(self.order_types, self.num_cities) # Dataframe of order types and locations
        self.data_inventory = self.generate_inventory_data(self.facility_indices, self.items, self.p_stock, self.inv_seed_value) # Generate inventory data for each facility and item
        self.df_inventory = pd.DataFrame(self.data_inventory) # Dataframe of inventory data (facility, product_id, potentially-in-stock)
        self.location_customers_by_facility_and_item = self.get_location_customers_by_facility_and_item(self.facility_indices, self.items, self.city_locations, self.facility_locations, self.df_inventory, self.num_items) # Set of cities served by each facility for each item
        self.expected_demand_k_i = self.calculate_expected_demand(self.facility_indices, self.items, self.location_customers_by_facility_and_item, self.order_types, self.adjusted_demand_distribution_by_type_by_location) # Expected demand for each facility and item
        self.safety_stock = self.calculate_safety_stock(self.facility_indices, self.items, self.T, self.expected_demand_k_i, self.CSL) # Safety stock for each facility and item
        # self.agg_adjusted_demand_distribution_by_type_by_location = self.aggregate_demand_distribution() # Aggregate (across time) demand distribution by type by location
        self.agg_adjusted_demand_distribution_by_type_by_location = self.aggregate_demand_distribution(self.T, self.adjusted_demand_distribution_by_type_by_location)
        self.reshape_adjusted_arrival_prob = self.reshape_modified_probs(self.adjusted_demand_distribution_by_type_by_location) # Reshape the adjusted demand distribution
        self.reshape_agg_adjusted_arrival_prob = self.reshape_probs(self.agg_adjusted_demand_distribution_by_type_by_location) # Reshape the aggregate demand distribution
        self.all_methods_location = self.find_all_order_methods_location(self.df_orders_location, self.facility_indices, self.safety_stock) # Find all methods for each (order,location)
        self.all_costs = self.calculate_all_costs(self.all_methods_location, self.fixed_costs, self.unit_costs) # Calculate costs for all methods
        self.all_indicators = self.calculate_all_indicators(self.all_methods_location, self.items, self.facility_indices) # Calculate indicators for all (i, k) pairs

    # Validate attributes
    def validate_attributes(self):
        necessary_attributes = ['num_items', 'n_max', 'n_0', 'p_stock', 'T', 'CSL', 'facilities_data', 
                                'cities_data', 'items', 'fulfillment_centers', 'facility_indices', 
                                'facility_locations', 'num_facilities', 'cities', 'city_indices', 
                                'city_locations', 'num_cities', 'population', 'total_population', 
                                'distances', 'fixed_costs', 'unit_costs', 'prob_order_size', 
                                'order_types', 'demand_distribution_by_type', 
                                'demand_distribution_by_type_by_location', 'adjusted_demand_distribution_by_type_by_location', 'df_orders', 
                                'df_orders_location', 'data_inventory', 'df_inventory', 
                                'location_customers_by_facility_and_item', 'expected_demand_k_i', 
                                'safety_stock', 'agg_adjusted_demand_distribution_by_type_by_location', 'reshape_adjusted_arrival_prob', 
                                'reshape_agg_adjusted_arrival_prob', 'all_methods_location', 'all_costs', 
                                'all_indicators']
        for attr in necessary_attributes:
            if not hasattr(self, attr):
                raise ValueError(f"Attribute {attr} has not been properly initialized.")

    # Items
    def generate_items_list(self, num_items):
        """Generates a list of item indices."""
        return list(range(num_items))

    # Load facility data
    def load_fulfillment_centers(self, file_path, use_single_facility=False):
        """Loads the fulfillment center data from a CSV file."""
        fulfillment_centers = pd.read_csv(file_path)
        if use_single_facility:
            fulfillment_centers = fulfillment_centers.iloc[:1]
        return fulfillment_centers

    # Generate facility indices and locations (longitude (y) and latitude (x))
    def generate_facility_indices_and_locations(self, fulfillment_centers_df):
        """Generates the indices and locations of fulfillment centers."""
        num_facilities = fulfillment_centers_df.shape[0]
        facility_indices = list(range(num_facilities))
        facility_locations = [(fulfillment_centers_df["Longitude"][r], fulfillment_centers_df["Latitude"][r]) for r in range(num_facilities)]
        return facility_indices, facility_locations

    # Load data about cities
    def load_cities(self, file_path, use_single_city=False):
        """Loads the city data from a CSV file."""
        cities_df = pd.read_csv(file_path)
        if use_single_city:
            cities_df = cities_df.iloc[:1]
        return cities_df

    # Generate city indices and locations (longitude (y) and latitude (x))
    def generate_city_indices_and_locations(self, cities_df):
        """Generates the indices and locations of cities."""
        num_cities = cities_df.shape[0]
        city_indices = list(range(num_cities))
        city_locations = [(cities_df["Longitude"][r], cities_df["Latitude"][r]) for r in range(num_cities)]
        return city_indices, city_locations

    # Get population data and total population
    def get_population_data(self, cities_df):
        population = cities_df["Population"]
        total_population = sum(population)
        return population, total_population
    
    # Plotting cities and facilities
    def plot_cities_and_facilities(self):
        city_x = [city_location[1] for city_location in self.city_locations]
        city_y = [city_location[0] for city_location in self.city_locations]
        city_indices = list(range(len(self.city_locations)))
        facility_x = [facility_location[1] for facility_location in self.facility_locations]
        facility_y = [facility_location[0] for facility_location in self.facility_locations]
        facility_indices = list(range(len(self.facility_locations)))
        fig, ax = plt.subplots()
        ax.scatter(city_x, city_y, c='b', marker='.')
        ax.scatter(facility_x, facility_y, c='r', marker='s')
        for i, txt in enumerate(city_indices):
            ax.annotate(txt, (city_x[i], city_y[i]))
        for i, txt in enumerate(facility_indices):
            ax.annotate(txt, (facility_x[i], facility_y[i]))
        plt.show()
        
    @staticmethod
    def haversine(lon1, lat1, lon2, lat2, inv):
        """
        Calculate the great circle distance in miles between two points on the Earth
        (specified in decimal degrees), given their inventory status.
        """
        if inv == 0: 
            return 10**6 # Return a large distance value if the facility has no inventory
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 3959 # Radius of Earth in miles
        return c * r
    
    # Calculate costs and distances
    def calculate_costs_and_distances(self, city_locations, facility_locations):
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
                dists[k][j] = self.haversine(facility_locations[k][0], facility_locations[k][1],
                                        city_locations[j][0], city_locations[j][1], inv=1)
                c_unit[k][j] = 0.423 + 0.000541 * dists[k][j]
                dist_max = max(dist_max, dists[k][j])
            # Routing to FC K+1 corresponds to not fulfilling an item
            c_fixed[num_facilities][j] = 2 * 8.759  # Twice the fixed cost
            c_unit[num_facilities][j] = 2 * (0.423 + 0.000541 * dist_max)  # Twice the variable cost of fulfilling from the farthest facility
        return dists, c_fixed, c_unit
    
    # Generate order types and respective probabilities of arrival
    @staticmethod
    def random_combination(iterable, r):
        """Random selection of r elements from iterable"""
        pool = tuple(iterable)
        n = len(pool)
        indices = sorted(random.sample(range(n), r))
        return tuple(pool[i] for i in indices)

    # Generate a probability distribution for the order size, including the empty order
    @staticmethod
    def generate_order_size_probabilities(n_max, seed_value):
        """Generate a probability distribution for the order size, including the empty order."""
        random.seed(seed_value)
        prob_size = random.sample(range(0, 1000), n_max + 1)
        sum_p = sum(prob_size)
        prob_order_size = [i / sum_p for i in prob_size]
        return prob_order_size

    # Generate odert types and a probability distribution by type for each order size n
    def generate_demand_distribution(self, n_max, n_0, prob_order_size, seed_value):
        random.seed(seed_value)
        np.random.seed(seed_value) # I have two sources of randomness and want to make sure they are reproducible
        order_types = [()]
        demand_distribution_by_type = [[prob_order_size[0]]] # order of size 0 equally likely across types
        
        for size in range(1, n_max + 1):
            max_combinations = int(comb(self.num_items, size))
            target_combinations = min(n_0, max_combinations)
            unique_orders = set()
            
            while len(unique_orders) < target_combinations:
                order = self.random_combination(range(self.num_items), size)
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
    
    # Scale the demand probabilities by location
    def scale_demand_probabilities_by_location(self, demand_distribution_by_type, population, total_population):
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
    
    def adjust_arrival_probabilities(self, T):
        """
        Adjust the arrival probabilities based on whether we are in the first or second half of fulfillment and the order size.
        Normalizes the probabilities across all order types and locations for each time period t.
        """
        # Initialize the structure to hold the adjusted probabilities
        adjusted_demand_distribution_by_type_by_location = []

        # Determine scaling factors based on the time period
        alpha_scale_first_half = 2 * self.alpha
        alpha_scale_second_half = 2 * (1 - self.alpha)

        # Loop over each time period
        for t in range(1, T + 1):
            temp_probs_by_time_period = []  # Temporary list for the adjusted probabilities in the current period by order size

            # Store all probabilities for normalization purposes
            all_probs_for_normalization = []

            # Loop over each order size
            for n, order_types_probs in enumerate(self.demand_distribution_by_type_by_location):
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
                            adjusted_prob = alpha_scale_first_half * prob if 1 <= t <= self.T // 2 else alpha_scale_second_half * prob
                        else:
                            # Scale probabilities for order size greater than 1
                            adjusted_prob = alpha_scale_second_half * prob if 1 <= t <= self.T // 2 else alpha_scale_first_half * prob
                        adjusted_probs_by_location.append(adjusted_prob)

                    temp_probs_by_order_type.extend(adjusted_probs_by_location)

                # Collect all probabilities for normalization
                all_probs_for_normalization.extend(temp_probs_by_order_type)

            # Normalize the collected probabilities so they sum to 1
            total_prob = sum(all_probs_for_normalization)
            normalized_probs = [prob / total_prob for prob in all_probs_for_normalization] if total_prob > 0 else all_probs_for_normalization

            # Re-distribute the normalized probabilities back to the original structure
            index = 0
            for n, order_types_probs in enumerate(self.demand_distribution_by_type_by_location):
                temp_probs_by_order_type = []
                for q, probs_by_location in enumerate(order_types_probs):
                    normalized_probs_by_location = normalized_probs[index:index+len(probs_by_location)]
                    index += len(probs_by_location)
                    temp_probs_by_order_type.append(normalized_probs_by_location)
                temp_probs_by_time_period.append(temp_probs_by_order_type)

            # Add the adjusted and normalized probabilities for the current period to the overall structure
            adjusted_demand_distribution_by_type_by_location.append(temp_probs_by_time_period)

        return adjusted_demand_distribution_by_type_by_location

    
    # def adjust_arrival_probabilities(self, T):
    #     """
    #     Adjust the arrival probabilities based on whether we are in the first or second half of fulfillment and the order size.
    #     """
    #     # Initialize the structure to hold the adjusted probabilities
    #     adjusted_demand_distribution_by_type_by_location = []

    #     # Determine scaling factors based on the time period
    #     alpha_scale_first_half = 2 * self.alpha
    #     alpha_scale_second_half = 2 * (1 - self.alpha)

    #     # Loop over each time period
    #     for t in range(1, T + 1):
    #         # Temporary list for the adjusted probabilities in the current period by order size
    #         temp_probs_by_time_period = []

    #         # Loop over each order size
    #         for n, order_types_probs in enumerate(self.demand_distribution_by_type_by_location):
    #             temp_probs_by_order_type = []  # Temp list for the current order size

    #             # Loop over each order type for the current order size
    #             for q, probs_by_location in enumerate(order_types_probs):
    #                 # Adjust the probabilities based on the order size and the time period
    #                 adjusted_probs_by_location = []
    #                 for j, prob in enumerate(probs_by_location):
    #                     if n == 0:
    #                         # Leave the zero order size probabilities as they are
    #                         adjusted_prob = prob
    #                     elif n == 1:
    #                         # Scale probabilities for order size of 1
    #                         adjusted_prob = alpha_scale_first_half * prob if 1 <= t <= self.T // 2 else alpha_scale_second_half * prob
    #                     else:
    #                         # Scale probabilities for order size greater than 1
    #                         adjusted_prob = alpha_scale_second_half * prob if 1 <= t <= self.T // 2 else alpha_scale_first_half * prob
    #                     adjusted_probs_by_location.append(adjusted_prob)

    #                 temp_probs_by_order_type.append(adjusted_probs_by_location)

    #             temp_probs_by_time_period.append(temp_probs_by_order_type)

    #         # Add the adjusted probabilities for the current period to the overall structure
    #         adjusted_demand_distribution_by_type_by_location.append(temp_probs_by_time_period)

    #     return adjusted_demand_distribution_by_type_by_location
    
    
    # Create dataframe of order types
    def create_orders_df(self, order_types):
        temp_lista = []
        for i in range(len(order_types)):
            temp_lista.append(order_types[i])
        df_orders = pd.DataFrame()
        df_orders['order'] = temp_lista
        return df_orders

    # Create dataframe of order types and locations
    def create_orders_location_df(self, order_types, num_cities):
        temp_lista_location = []
        for i in range(len(order_types)):
            for j in range(num_cities):
                temp_lista_location.append((order_types[i],j))
        df_orders_location = pd.DataFrame()
        df_orders_location['(order,location)'] = temp_lista_location
        return df_orders_location
    
    # INVENTORY DATA for each facility k and item i
    def generate_inventory_data(self, facility_indices, items, p_stock, seed_value):
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
    def get_location_customers_by_facility_and_item(self, facility_indices, items, city_locations, facility_locations, df_inventory, num_items):
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
                    distance_closest_facility_to_j = round(min(self.haversine(facility_locations[k_1][0], facility_locations[k_1][1], 
                                                                        city_locations[j][0], city_locations[j][1], 
                                                                        df_inventory["potentially-in-stock"][k_1*num_items+i]) for k_1 in facility_indices),4)
                    # Calculate the distance from facility k to city j
                    distance_k_j = round(self.haversine(facility_locations[k][0], facility_locations[k][1], 
                                                city_locations[j][0], city_locations[j][1], df_inventory["potentially-in-stock"][k*num_items+i]),4)
                    if distance_k_j == distance_closest_facility_to_j:
                        temp_i.append(j)
                temp_k.append(temp_i)
            location_customers_by_facility_and_item.append(temp_k)
        return location_customers_by_facility_and_item
   
    # Calculate the total incoming expected demand for each facility and item 
    def calculate_expected_demand(self, facility_indices, items, location_customers_by_facility_and_item, order_types, adjusted_demand_distribution_by_type_by_location):
        """
        Calculate the total incoming expected demand for each facility and item.
        """
        expected_demand_k_i = []
        for k in facility_indices:
            for i in items:
                prob = 0
                # Iterate over cities served by the facility for the current item
                for j in location_customers_by_facility_and_item[k][i]:
                    # Iterate over demand types
                    for order in order_types:
                        if i in order:
                            index_order = order_types.index(order)
                            # Calculate average probability of the order occurring at city j over all time periods
                            avg_prob = sum(adjusted_demand_distribution_by_type_by_location[t][len(order)][index_order-self.n_0*(len(order)-1)-1][j] for t in range(len(adjusted_demand_distribution_by_type_by_location))) / len(adjusted_demand_distribution_by_type_by_location)
                            prob += avg_prob
                expected_demand_k_i.append(prob)
        return expected_demand_k_i

    # Calculate safety stock for each facility k and item i
    # def calculate_safety_stock(self, facility_indices, items, T, expected_demand_k_i, CSL, modified = 0):
    #     """
    #     Calculate inventory level. No safety stock if modified = 0
    #     """
    #     mu = []
    #     sd = []
    #     for k in facility_indices:
    #         for i in items:
    #             mu.append(T * expected_demand_k_i[k*len(items)+i])
    #             sd.append(np.sqrt(T * expected_demand_k_i[k*len(items)+i] * (1 - expected_demand_k_i[k*len(items)+i])))
    #     S = []
    #     for k in facility_indices:
    #         S_k = []
    #         for i in items:
    #             S_k.append(np.ceil(mu[k*len(items)+i] + modified * sps.norm.ppf(CSL, loc=0, scale=1) * sd[k*len(items)+i]))
    #         S.append(S_k)
    #     return S
    
    def calculate_safety_stock(self, facility_indices, items, T, expected_demand_k_i, CSL, modified = 0):
        """
        Calculate inventory level. No safety stock if modified = 0
        """
        mu = []
        for k in facility_indices:
            for i in items:
                mu.append(T * expected_demand_k_i[k*len(items)+i])
        S = []
        for k in facility_indices:
            S_k = []
            for i in items:
                S_k.append(np.ceil(mu[k*len(items)+i]))
            S.append(S_k)
        return S
    
    # METHODS FOR ORDER TYPES
    @staticmethod
    def powerset(iterable):
        """Return the powerset of the given iterable."""
        s = list(iterable)
        # Generate a powerset (all possible subsets) of the input iterable by iterating through all possible combinations
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    # Find facilities containing a given item
    def facility_set_per_item(self, item_id, facilities, inventories):
        """Find facilities containing the specified item."""
        available_facilities = []
        for k in facilities:
            # If the facility has the item in stock, add it to the list of available facilities
            if inventories[k][item_id] > 0:
                available_facilities.append(k)
        return available_facilities

    # Find all possible methods to fulfill (potentially partially) an order
    def find_all_methods(self, order, facilities, inventories):
        """Find all possible methods to fulfill (potentially partially) an order."""
        all_methods = [[()]]  # Include the "do-nothing" method
        # Iterate through the items in the order
        for item in order:
            # Find facilities that have the item in stock
            facility_item = self.facility_set_per_item(item, facilities, inventories)
            # Add each facility-item pair as an individual method
            for k in facility_item:
                all_methods.append([(item, k)])
        # Calculate the powerset of the order to get all possible subsets
        all_subsets_order = list(self.powerset(order))
        for element in all_subsets_order:
            # Skip empty subsets
            if not element:
                continue
            # Get facilities that have items in the subset in stock
            facility_sets = [self.facility_set_per_item(i, facilities, inventories) for i in element]
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
    
    # Aggregate demand distribution over time (this does not change because of the transformation on the arrival probabilities)
    # def aggregate_demand_distribution(self):
    #     # In the next line, demand_distribution_by_type_by_location is the ORIGINAL ARRIVAL PROBAB
    #     return [[[element * self.T for element in row] for row in submatrix] for submatrix in self.demand_distribution_by_type_by_location]

    def aggregate_demand_distribution(self, T, adjusted_demand_distribution_by_type_by_location):
        # Extract the first and last time period distributions
        first_period_distribution = adjusted_demand_distribution_by_type_by_location[0]
        last_period_distribution = adjusted_demand_distribution_by_type_by_location[-1]

        # Calculate scaling factors based on T
        scale_factor = T / 2

        # Initialize the structure for the new distribution
        aggregate_demand_distribution = []

        # Iterate over the order sizes
        for n, (first_order_types, last_order_types) in enumerate(zip(first_period_distribution, last_period_distribution)):
            new_order_type_probs = []

            # Iterate over the order types for the current order size
            for q, (first_probs_by_location, last_probs_by_location) in enumerate(zip(first_order_types, last_order_types)):
                new_probs_by_location = []

                # Iterate over each location for the current order type and size
                for first_prob, last_prob in zip(first_probs_by_location, last_probs_by_location):
                    # Calculate the new probability based on the formula provided
                    new_prob = scale_factor * first_prob + scale_factor * last_prob
                    new_probs_by_location.append(new_prob)

                new_order_type_probs.append(new_probs_by_location)
            
            aggregate_demand_distribution.append(new_order_type_probs)

        return aggregate_demand_distribution

    # Reshape a nested (3D) distribution
    def reshape_probs(self, nested_distribution):
        """reshape a nested (3D) distribution"""
        reshape_arrival_prob = []
        for i in nested_distribution:
            for k in i:
                for j in k:
                    reshape_arrival_prob.append(j)
        return reshape_arrival_prob
    
    # Reshape a nested (4D) distribution
    def reshape_modified_probs(self, nested_distribution):
        reshape_arrival_prob = []
        for t in range(len(nested_distribution)):
            reshape_arrival_prob.append(self.reshape_probs(nested_distribution[t]))
        return reshape_arrival_prob
    
    # For each (order,location) find all methods. Each elements is a list of methods, with each method being a list of (i,k) tuples
    def find_all_order_methods_location(self, df_orders_location, facility_indices, safety_stock):
        """For each (order,location) find all methods-location."""
        all_methods_location = []
        for order in df_orders_location['(order,location)']:
            order_only = order[0]  # extract only the order part from the tuple
            order_location = order[1]
            all_methods_location.append({'methods':self.find_all_methods(order_only, facility_indices, safety_stock), 'order': order_only, 'location': order_location, 'size': len(order_only)})
        return all_methods_location
    
    # Calculate costs for all methods
    def calculate_all_costs(self, all_methods_location, c_fixed, c_unit):
        """
        Calculate costs for all methods.
        """
        all_costs = [[0]]*self.num_cities # cost of empty order (from anywhere) is zero
        for methods_location in all_methods_location[self.num_cities:]: # exclude empty orders because we already added the cost of empty orders
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
    
    # Calculate indicator (i,k) in m
    def calculate_all_indicators_i_k(self, all_methods_location, i_k_pair):
        """Indicator (i,k) in m."""
        all_indicators = []
        for methods_location in all_methods_location:
            indicators = []
            for method in methods_location['methods']:
                if i_k_pair in method:
                    indicators.append(1)
                else:
                    indicators.append(0)
            all_indicators.append(indicators)
        return all_indicators

    # Calculate indicators for all (i, k) pairs
    def calculate_all_indicators(self, all_methods_location, items, facilities):
        """Calculate indicators for all (i, k) pairs."""
        all_indicators = {}
        for i in items:
            for k in facilities:
                i_k_pair = (i, k)
                all_indicators[i_k_pair] = self.calculate_all_indicators_i_k(all_methods_location, i_k_pair)
        return all_indicators


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    