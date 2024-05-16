# cluster_fulfillment_policy_notidentical_arrival_probs.py

import numpy as np
import random
from order_fulfillment_network import OrderFulfillment
from magician_problem import MagicianProblem

class FulfillmentPolicy:
    def __init__(self, order_fulfillment: OrderFulfillment, LP_solution, methods, sizes, consumption_probability_lists, get_optimization_results):
        # Order_fulfillment object passed as parameter
        self.order_fulfillment = order_fulfillment
        # Validate that all necessary attributes have been initialized
        self.order_fulfillment.validate_attributes()
        # Inventory
        self.S = self.order_fulfillment.safety_stock
    
        # LP solution, methods, sizes, and consumption probabilities are passed as parameters
        self.LP_solution = LP_solution
        self.methods = methods
        self.sizes = sizes
        self.consumption_probability_lists = consumption_probability_lists
        
        # Obtain optimization results
        self.get_optimization_results = get_optimization_results
        
    # Function used to sample an order arrival index (reshape_arrival_prob will be used for this function)
    def sample_order_index(self, probabilities):
        indices = list(range(len(probabilities)))
        sample = random.choices(indices, weights=probabilities, k=1)[0]
        return sample

    # Function used to multiply each value of a dictionary by a constant (in order to define method distribution conditional on the sampled order)
    def dict_mult(self, dict, constant):
        """Multiply each value of a dictionary by a constant"""
        new_dict = {}
        for key in dict:
            new_dict[key] = dict[key]*constant
        return(new_dict)

    # Function used to sample a method index for the sampled order
    def sample_dict_key(self, dictionary):
        """Sample from values of a dictionary"""
        keys = list(dictionary.keys())
        probabilities = list(dictionary.values())
        sample = random.choices(keys, weights=probabilities, k=1)[0]
        return sample

    # Implementation of probabilistic fulfillment
    def probabilistic_fulfillment(self, t):
        # sampled order
        sampled_order = self.sample_order_index(self.order_fulfillment.reshape_probs(self.order_fulfillment.adjusted_demand_distribution_by_type_by_location[t]))
        # method distribution conditional on sampled order
        sample_method_dict = self.dict_mult(self.LP_solution[sampled_order], 1/self.order_fulfillment.reshape_agg_adjusted_arrival_prob[sampled_order])
        # sample index for a method for the sampled order
        sampled_value = self.sample_dict_key(sample_method_dict)
        # sample a method
        sampled_method = self.methods[sampled_order][sampled_value]
        return(sampled_order, sampled_method, sampled_value)
    
    def solve_cdfs_magician_problems(self, conservative_prob):
        cdfs_magician_problems = {}
        for pair_i_k in self.order_fulfillment.all_indicators:
            # Get the breaking wand probabilities for (i,k) and the gamma value
            breaking_wand_probabilities_ik = self.consumption_probability_lists[pair_i_k]
            gamma_ik = 1-conservative_prob/np.sqrt(self.order_fulfillment.safety_stock[pair_i_k[1]][pair_i_k[0]]+3)
            # Create magician instance for (i,k) and compute all cdfs
            magician_problem = MagicianProblem(breaking_wand_probabilities_ik, gamma_ik, self.order_fulfillment)
            magician_problem.compute_all_CDFs()
            cdfs_magician_ik = magician_problem.all_cdfs
            cdfs_magician_problems[pair_i_k] = cdfs_magician_ik
        return cdfs_magician_problems
    
    def solve_magician_problems(self, conservative_prob, cdfs_magician_problems):
        magician_problems = {}
        for pair_i_k in self.order_fulfillment.all_indicators:
            # Create a magician problem
            magician_problems[pair_i_k] = {}
            magician_problems[pair_i_k]['breaking_wand_probabilities'] = self.consumption_probability_lists[pair_i_k]
            magician_problems[pair_i_k]['gamma'] = 1-conservative_prob/np.sqrt(self.order_fulfillment.safety_stock[pair_i_k[1]][pair_i_k[0]]+3)
            # if sum(magician_problems[pair_i_k]['breaking_wand_probabilities']) > self.order_fulfillment.safety_stock[pair_i_k[1]][pair_i_k[0]]:
            #     print('expected number of broken wands larger than the available ones')
            # Create magician instance for (i,k) and check whether they open
            magician_problem = MagicianProblem(magician_problems[pair_i_k]['breaking_wand_probabilities'], magician_problems[pair_i_k]['gamma'], self.order_fulfillment)
            theta, open_list, prob_rand = magician_problem.solve_withcdfs(cdfs_magician_problems[pair_i_k])
            magician_problems[pair_i_k]['theta'] = theta
            magician_problems[pair_i_k]['open_list'] = open_list
            magician_problems[pair_i_k]['prob_rand'] = prob_rand
        return magician_problems
    
    def initialize_inventory_consumption(self):
        inventory_consumption = {}
        for pair_i_k in self.order_fulfillment.all_indicators:
            inventory_consumption[pair_i_k] = 0
        return inventory_consumption
    
    def new_fulfillment_policy_end_modified(self, inventory_consumption, magician_problems, seed_value):
        """ New magician-based gamma-conservative acceptance policy """
        random.seed(seed_value)  # Used for reproducibility

        def check_inventory_availability(pair_i_k):
            """ Check if there is sufficient inventory available """
            return inventory_consumption[pair_i_k] < self.S[pair_i_k[1]][pair_i_k[0]]

        def check_magician_open(pair_i_k, t):
            """ Check if magician problem opens the box or not """
            opening_index = inventory_consumption[pair_i_k]
            return magician_problems[pair_i_k]['open_list'][t][opening_index] == 1 # magician opens the box
        
        cities_set = set(range(self.order_fulfillment.num_cities))
        
        sampled_orders = []
        sampled_orders_index = []
        sampled_methods = []
        accepts_decisions = []
        fulfillment_costs = []

        for t in range(self.order_fulfillment.T):
            
            sampled_order, sampled_method, method_index = self.probabilistic_fulfillment(t)
            accept = True
            cost = 0

            if sampled_order in cities_set or sampled_method == [()]:
                accept = False
                cost = self.order_fulfillment.all_costs[sampled_order][method_index] # Cost of do-nothing or cost of empty order (which is zero)
            else:
                order_items, j = self.order_fulfillment.df_orders_location.iloc[sampled_order][0]
                fc_used = [False] * (self.order_fulfillment.num_facilities + 1) # Initialized to calculate fixed cost

                # If the method is already partial, add costs for not fulfilling the remaining items
                if len(sampled_method) != len(order_items):
                        fc_used[-1] = True # We will use the dummy facility (we will add cost later)
                        cost += self.order_fulfillment.unit_costs[-1][j] * (len(order_items) - len(sampled_method)) # Add variable cost for each unfulfilled item

                # For pairs (i,k) in the method, use the magicians
                count_magician_i_k_accepts = 0
                for pair_i_k in sampled_method:
                    i, k = pair_i_k
                    if t == self.order_fulfillment.T-1 and check_inventory_availability(pair_i_k):
                        count_magician_i_k_accepts += 1
                        inventory_consumption[pair_i_k] += 1
                    elif check_inventory_availability(pair_i_k) and check_magician_open(pair_i_k, t):
                        count_magician_i_k_accepts += 1
                        inventory_consumption[pair_i_k] += 1
                    else:
                        k = self.order_fulfillment.num_facilities # index of dummy facility
                    fc_used[k] = True
                    cost += self.order_fulfillment.unit_costs[k][j]  # unit cost does not change across items
                
                for k in range(self.order_fulfillment.num_facilities + 1):
                    if fc_used[k]:
                        cost += self.order_fulfillment.fixed_costs[k][j]
                
                if count_magician_i_k_accepts != len(sampled_method):
                    accept = False # If some of the magicians do not accept, then we do not accept the method
                
            # Append values to lists
            sampled_orders_index.append(sampled_order)
            sampled_orders.append(self.order_fulfillment.df_orders_location.iloc[sampled_order][0])
            sampled_methods.append(sampled_method)
            accepts_decisions.append(accept)
            fulfillment_costs.append(cost)

        return sampled_orders_index, sampled_orders, sampled_methods, accepts_decisions, fulfillment_costs
    
    def always_accept_policy(self, inventory_consumption, seed_value):
        """ Always accept methods as long as you have enough inventory """
        random.seed(seed_value)
        
        def check_inventory_availability(pair_i_k):
            """ Check if there's sufficient inventory available """
            return inventory_consumption[pair_i_k] < self.S[pair_i_k[1]][pair_i_k[0]]  # If this condition is not met, we cannot accept a method

        # Convert num_cities to a set for efficient membership checking
        cities_set = set(range(self.order_fulfillment.num_cities))

        sampled_orders_aa = []
        sampled_orders_index_aa = []
        sampled_methods_aa = []
        accepts_decisions_aa = []
        fulfillment_costs_aa = []
        
        for t in range(self.order_fulfillment.T):
            sampled_order_aa, sampled_method_aa, method_index_aa = self.probabilistic_fulfillment(t)
            accept_aa = True
            cost = 0
            
            # Check for special conditions (empty order or do-nothing method)
            if sampled_order_aa in cities_set or sampled_method_aa == [()]:  # Check if the order is empty or the method is do-nothing
                accept_aa = False
                cost = self.order_fulfillment.all_costs[sampled_order_aa][method_index_aa]
            else:
                order_items, j = self.order_fulfillment.df_orders_location.iloc[sampled_order_aa][0]
                fc_used = [False] * (self.order_fulfillment.num_facilities + 1)  # Initialized to calculate fixed cost

                # If the method is already partial, add costs for not fulfilling the remaining items
                if len(sampled_method_aa) != len(order_items):
                    fc_used[-1] = True  # We will use the dummy facility (we will add cost later)
                    cost += self.order_fulfillment.unit_costs[-1][j] * (len(order_items) - len(sampled_method_aa))  # Add variable cost for each unfulfilled item

                for pair_i_k in sampled_method_aa:
                    i, k = pair_i_k
                    if check_inventory_availability(pair_i_k):
                        inventory_consumption[pair_i_k] += 1
                    else:
                        k = self.order_fulfillment.num_facilities  # Index of dummy facility
                    fc_used[k] = True
                    cost += self.order_fulfillment.unit_costs[k][j]  # Unit cost does not change across items
                
                for k in range(self.order_fulfillment.num_facilities + 1):
                    if fc_used[k]:
                        cost += self.order_fulfillment.fixed_costs[k][j]

            # Append values to result lists
            sampled_orders_index_aa.append(sampled_order_aa)
            sampled_orders_aa.append(self.order_fulfillment.df_orders_location.iloc[sampled_order_aa][0])
            sampled_methods_aa.append(sampled_method_aa)
            accepts_decisions_aa.append(accept_aa)
            fulfillment_costs_aa.append(cost)

        return sampled_orders_index_aa, sampled_orders_aa, sampled_methods_aa, accepts_decisions_aa, fulfillment_costs_aa

    def check_consistency(self, inventory_consumption):
        for k in range(self.order_fulfillment.num_facilities):
            for i in range(self.order_fulfillment.num_items):
                if inventory_consumption[(i, k)] > self.order_fulfillment.safety_stock[k][i]:
                    print('Warning: consumption of resources larger than inventory')
                    print(inventory_consumption[(i, k)], self.order_fulfillment.safety_stock[k][i])
    




