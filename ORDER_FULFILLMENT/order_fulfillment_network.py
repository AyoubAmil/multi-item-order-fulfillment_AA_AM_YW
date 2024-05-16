import data_preparation
from demand_distribution import generate_order_size_probabilities, scale_demand_probabilities_by_location, generate_demand_distribution, adjust_arrival_probabilities, aggregate_demand_distribution
from inventory import generate_inventory_data, get_location_customers_by_facility_and_item, calculate_expected_demand, calculate_safety_stock
from orders import create_orders_df, create_orders_location_df
from costs import calculate_costs_and_distances, calculate_all_costs
import methods 
import utilities
import pandas as pd


class OrderFulfillment:
    def __init__(self, num_items=20, n_max=5, n_0=5, p_stock=0.75, T=10**3, facilities_data="Data/fulfillment_centers_warmup.csv", cities_data="Data/cities_warmup.csv", prob_seed_value=1, order_seed_value=1, inv_seed_value=1, alpha=0.5):
        
        # Seed values
        self.prob_seed_value = prob_seed_value # Used for reproducibility of arrival probabilities
        self.order_seed_value = order_seed_value # Used for reproducibility of order types
        self.inv_seed_value = inv_seed_value # Used for reproducibility of inventories
        
        # Parameters of the class
        self.num_items = num_items # Number of items
        self.n_max = n_max # Maximum order size
        self.n_0 = n_0 # Number of order types for each order size
        self.p_stock = p_stock # Probability that an item is in stock at a facility
        self.T = T # Time horizon
        self.facilities_data = facilities_data # Path to the CSV file containing the facility data
        self.cities_data = cities_data # Path to the CSV file containing the city data
        self.alpha = alpha # Parameter for the transformation of the arrival probabilities. Alpha = 0.5 corresponds to the iid case, which is what we use in the simulation
        
        # Data Preparation
        self.items = data_preparation.generate_items_list(self.num_items) # List of item indices
        self.fulfillment_centers = data_preparation.load_fulfillment_centers(self.facilities_data, use_single_facility=False) # Dataframe of fulfillment centers
        self.facility_indices, self.facility_locations = data_preparation.generate_facility_indices_and_locations(self.fulfillment_centers) # List of fulfillment center indices and locations
        self.num_facilities = len(self.facility_indices) # Number of fulfillment centers
        self.cities = data_preparation.load_cities(self.cities_data, use_single_city=False) # Dataframe of cities
        self.city_indices, self.city_locations = data_preparation.generate_city_indices_and_locations(self.cities) # List of city indices and locations
        self.num_cities = len(self.city_indices) # Number of cities
        self.population, self.total_population = data_preparation.get_population_data(self.cities) # List of city populations and total population
        
        # Demand distribution orders
        self.prob_order_size = generate_order_size_probabilities(self.n_max, self.prob_seed_value) # Probability distribution for the order size, including the empty order
        self.order_types, self.demand_distribution_by_type = generate_demand_distribution(self.num_items, self.n_max, self.n_0, self.prob_order_size, self.order_seed_value) # Probability distribution by type for each order size n
        self.demand_distribution_by_type_by_location = scale_demand_probabilities_by_location(self.demand_distribution_by_type, self.population, self.total_population) # Demand distribution by type by location
        self.adjusted_demand_distribution_by_type_by_location = adjust_arrival_probabilities(self.T, self.alpha, self.demand_distribution_by_type_by_location) # Adjust the arrival probabilities based on whether we are in the first or second half of fulfillment and the order size
        self.agg_adjusted_demand_distribution_by_type_by_location = aggregate_demand_distribution(self.adjusted_demand_distribution_by_type_by_location)
        self.reshape_adjusted_arrival_prob = utilities.reshape_modified_probs(self.adjusted_demand_distribution_by_type_by_location) # Reshape the adjusted demand distribution
        self.reshape_agg_adjusted_arrival_prob = utilities.reshape_probs(self.agg_adjusted_demand_distribution_by_type_by_location) # Reshape the aggregate demand distribution
        
        # Orders
        self.df_orders = create_orders_df(self.order_types) # Dataframe of order types
        self.df_orders_location = create_orders_location_df(self.order_types, self.num_cities) # Dataframe of order types and locations
        
        # Inventory
        self.data_inventory = generate_inventory_data(self.facility_indices, self.items, self.p_stock, self.inv_seed_value) # Generate inventory data for each facility and item
        self.df_inventory = pd.DataFrame(self.data_inventory) # Dataframe of inventory data (facility, product_id, potentially-in-stock)
        self.location_customers_by_facility_and_item = get_location_customers_by_facility_and_item(self.facility_indices, self.items, self.city_locations, self.facility_locations, self.df_inventory, self.num_items) # Set of cities served by each facility for each item
        self.expected_demand_k_i = calculate_expected_demand(self.n_0, self.facility_indices, self.items, self.location_customers_by_facility_and_item, self.order_types, self.adjusted_demand_distribution_by_type_by_location) # Expected demand for each facility and item
        self.safety_stock = calculate_safety_stock(self.facility_indices, self.items, self.T, self.expected_demand_k_i) # Safety stock for each facility and item
        
        # Methods
        self.all_methods_location = methods.find_all_order_methods_location(self.df_orders_location, self.facility_indices, self.safety_stock) # Find all methods for each (order,location)
        self.all_indicators = methods.calculate_all_indicators(self.all_methods_location, self.items, self.facility_indices, self.safety_stock) # Calculate indicators for all (i, k) pairs
        
        # Costs
        self.distances, self.fixed_costs, self.unit_costs = calculate_costs_and_distances(self.city_locations, self.facility_locations) # Costs and distances between cities and fulfillment centers
        self.all_costs = calculate_all_costs(self.num_cities, self.all_methods_location, self.fixed_costs, self.unit_costs) # Calculate costs for all methods
        

    



