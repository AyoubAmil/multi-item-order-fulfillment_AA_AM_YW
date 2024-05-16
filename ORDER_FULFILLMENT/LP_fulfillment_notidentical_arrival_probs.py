# LP_fulfillment.py

import time
from order_fulfillment_network import OrderFulfillment
from gurobipy import GRB
from gurobipy import *
import gurobipy as gp

class SolvingLP:
    def __init__(self, order_fulfillment: OrderFulfillment):
        self.T = order_fulfillment.T  # Time horizon
        self.Q = order_fulfillment.df_orders_location # Order types (items, location)
        self.M = order_fulfillment.all_methods_location  # For each order type return dictionary with methods, location and size of the order
        self.p = order_fulfillment.reshape_agg_adjusted_arrival_prob  # Arrival probabilities for each order type (aggregated over time)
        self.S = order_fulfillment.safety_stock  # Inventory for each (i, k) pair
        self.c = order_fulfillment.all_costs  # Return the cost of each method
        self.all_indicators = order_fulfillment.all_indicators # Return the indicators $(i,k) \in m$ for each (i, k) pair that has inventory
        self.alpha = order_fulfillment.alpha
        self.reshape_adjusted_arrival_prob = order_fulfillment.reshape_adjusted_arrival_prob
        self.reshape_agg_adjusted_arrival_prob = order_fulfillment.reshape_agg_adjusted_arrival_prob

        self.solved_model = None
        self.model_solution = None

    def optimize_LP_relaxation(self):
        """
        Solve our LP relaxation.
        """
        start_time_init = time.time()
        
        model = gp.Model("LP relaxation")
        # Create variables
        x = {}
        methods = {}
        sizes = {}
        for q in range(len(self.Q)):
            # keys are indices for different order types and each value is a list of Gurobi variables
            x[q] = model.addVars(len(self.M[q]['methods']), vtype=GRB.CONTINUOUS, lb=0.0, ub=self.T, name=f"x_{q}")
            methods[q] = self.M[q]['methods']
            sizes[q] = self.M[q]['size']
        # Set objective
        model.setObjective(gp.quicksum(self.c[q][m]*x[q][m] for q in range(len(self.Q)) for m in range(len(self.M[q]['methods']))), GRB.MINIMIZE)
        # Add constraints
        model.addConstrs((gp.quicksum(x[q][m] for m in range(len(self.M[q]['methods']))) == self.p[q] 
                                    for q in range(len(self.Q))), name="constraint_1")

        # model.addConstrs((gp.quicksum(x[q][m] * self.all_indicators[key][q][m] for q in range(len(self.all_indicators[key])) for m in range(len(self.all_indicators[key][q]))) <= self.S[key[1]][key[0]] 
        #                           for key in self.all_indicators), name="constraint_2")
        
        model.addConstrs((gp.quicksum(x[q][m] for q in self.all_indicators[key] for m in self.all_indicators[key][q])
                  <= self.S[key[1]][key[0]] for key in self.all_indicators),
                 name="constraint_2")

        end_time_init = time.time()
        initialization_duration = end_time_init - start_time_init
        
        # Turn off the Gurobi output with model.setParam("OutputFlag", 0)
        model.setParam("OutputFlag", 1)
        
        # # Specify the Dual Simplex method
        # model.setParam("Method", 1)
        # Specify the Primal Simplex method
        model.setParam("Method", 0)
        
        # Start timing for optimization
        start_time_opt = time.time()
        
        # Solve model
        model.optimize()
        
        # End timing for optimization
        end_time_opt = time.time()
        optimization_duration = end_time_opt - start_time_opt
        
        # Check if a feasible solution is found
        if model.status == GRB.OPTIMAL:
            # Store the solved model
            self.solved_model = model
            # Get the solution
            x_sol = {q: {m: var.x for m, var in enumerate(x[q].values())} for q in x}
            # Store the solution
            self.model_solution = x_sol
            
            # Collecting optimization results
            num_vars = model.NumVars
            num_constrs = model.NumConstrs
            optimal_value = model.getObjective().getValue()
            
            return x_sol, methods, sizes, num_vars, num_constrs, optimal_value, initialization_duration, optimization_duration
        else:
            return None, None, None, None, None, None, initialization_duration, optimization_duration

    def get_optimization_results(self):
        if self.solved_model is not None and self.model_solution is not None:
            model = self.solved_model
            x = self.model_solution

            # Collecting optimization results
            num_vars = model.NumVars
            num_constrs = model.NumConstrs
            optimal_value = model.getObjective().getValue()

            # Extracting variable values
            x_values = {q: {m: x[q][m] for m in x[q]} for q in x}

            return {
                "x_values": x_values, 
                "num_vars": num_vars, 
                "num_constrs": num_constrs, 
                "optimal_value": optimal_value
            }
        else:
            print("Model has not been solved yet or no optimal solution was found.")
            return None

    def calculate_probabilities_of_consumption(self, LP_solution):
        """Calculate time-dependent probability of consumption at time t for each pair (i, k), adjusted by demand distributions."""
        consumption_probability_lists = {}

        # Flatten agg_adjusted_demand_distribution_by_type_by_location
        agg_distribution_flat = self.reshape_agg_adjusted_arrival_prob

        # Loop through all (i, k) pairs
        for i_k_pair, indicators_by_q in self.all_indicators.items():
            # Initialize an empty list to store probabilities for each time t
            consumption_probability_list = []

            # Loop through all time periods
            for t in range(1, self.T + 1):
                
                probability_t = 0

                # Flatten the distribution for time t to match structure of agg_distribution_flat
                distribution_t_flat = self.reshape_adjusted_arrival_prob[t-1]

                # Loop through each method group `q` that has relevant method indices for (i, k)
                for q, relevant_ms in indicators_by_q.items():
                    
                    adjusted_prob_t = distribution_t_flat[q]
                    agg_prob = agg_distribution_flat[q]
                    
                    scaled_solution = [LP_solution[q][m] * (adjusted_prob_t / agg_prob) for m in relevant_ms]
                    
                    probability_t += sum(scaled_solution)
            
                consumption_probability_list.append(probability_t)

            # Save the probability list for the (i, k) pair
            consumption_probability_lists[i_k_pair] = consumption_probability_list

        return consumption_probability_lists





