# LP_fulfillment.py

from order_fulfillment_environment import OrderFulfillment
from gurobipy import GRB
from gurobipy import *
import gurobipy as gp

class SolvingLP:
    def __init__(self, order_fulfillment: OrderFulfillment):
        self.T = order_fulfillment.T  # Time horizon
        self.Q = order_fulfillment.df_orders_location # Order types (items, location)
        self.M = order_fulfillment.all_methods_location  # For each order type return dictionary with methods, location and size of the order
        self.p = order_fulfillment.reshape_agg_arrival_prob  # Arrival probabilities for each order type (aggregated over time)
        self.S = order_fulfillment.safety_stock  # Inventory for each (i, k) pair
        self.c = order_fulfillment.all_costs  # Return the cost of each method
        self.all_indicators = order_fulfillment.all_indicators # Return the indicators $(i,k) \in m$ for each (i, k) pair

        self.solved_model = None
        self.model_solution = None

    def optimize_LP_relaxation(self):
        """
        Solve our LP relaxation.
        """
        model = gp.Model("LP relaxation")
        # Create variables
        x = {}
        methods = {}
        for q in range(len(self.Q)):
            # keys are indices for different order types and each value is a list of Gurobi variables
            x[q] = model.addVars(len(self.M[q]['methods']), vtype=GRB.CONTINUOUS, lb=0.0, ub=self.T, name=f"x_{q}")
            methods[q] = self.M[q]['methods']
        # Set objective
        model.setObjective(gp.quicksum(self.c[q][m]*x[q][m] for q in range(len(self.Q)) for m in range(len(self.M[q]['methods']))), GRB.MINIMIZE)
        # Add constraints
        model.addConstrs((gp.quicksum(x[q][m] for m in range(len(self.M[q]['methods']))) == self.p[q] 
                                    for q in range(len(self.Q))), name="constraint_1")

        model.addConstrs((gp.quicksum(x[q][m] * self.all_indicators[key][q][m] for q in range(len(self.all_indicators[key])) for m in range(len(self.all_indicators[key][q]))) <= self.S[key[1]][key[0]] 
                                  for key in self.all_indicators), name="constraint_2")
        
        # Turn off the Gurobi output
        model.setParam("OutputFlag", 0)
        
        # Solve model
        model.optimize()
        # Check if a feasible solution is found
        if model.status == GRB.OPTIMAL:
            # Store the solved model
            self.solved_model = model
            # Get the solution
            x_sol = {q: {m: var.x for m, var in enumerate(x[q].values())} for q in x}
            # Store the solution
            self.model_solution = x_sol
            return x_sol, methods
        else:
            return None
    
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
        """Calculate probability of consumption at time t for each pair (i,k). Used in the implementation of each magician."""
        probabilities = {}
        # Loop through all (i, k) pairs
        for i_k_pair in self.all_indicators:
            # Get the indicators for the (i, k) pair
            indicators_i_k = self.all_indicators[i_k_pair]
            # Calculate solutions for methods that contain (i, k)
            solution_methods_i_k = [LP_solution[q][method_index] for q in range(len(indicators_i_k)) for method_index, method_value in enumerate(indicators_i_k[q]) if method_value == 1]
            # Probability of consumption at time t
            probability = sum(solution_methods_i_k) / self.T
            # Save the total quantity for the (i, k) pair
            probabilities[i_k_pair] = probability
        return probabilities
    
    def generate_consumption_probability_lists(self, probabilities):
        """Generate consumption probability lists for each pair (i,k)."""
        consumption_probability_lists = {}
        for i_k_pair in self.all_indicators:
            consumption_probability_lists[i_k_pair] = [probabilities[i_k_pair] for _ in range(self.T)]
        return consumption_probability_lists


















