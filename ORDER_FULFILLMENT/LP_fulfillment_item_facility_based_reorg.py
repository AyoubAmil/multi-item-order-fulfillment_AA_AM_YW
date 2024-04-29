# LP_fulfillment_item_facility_based.py

import time
import numpy as np
from order_fulfillment_environment_notidentical_arrival_probs import OrderFulfillment
import gurobipy as gp
from gurobipy import GRB, Model

class ItemFacilityLPSolver:
    def __init__(self, order_fulfillment: OrderFulfillment):
        self.order_fulfillment = order_fulfillment

    def optimize(self):
        start_time_init = time.time()

        # Create a new model
        model = Model("ItemFacilityLP")

        # Dimensions and data extraction
        Q, K, I, J = self._get_dimensions()
        c_fixed, c_unit = self._get_costs()
        order_types, lambda_, S = self._get_order_types_probs_and_safety_stocks()

        # Variables
        u = model.addVars(Q, K + 1, I, J, vtype=GRB.CONTINUOUS, lb=0, name="u")
        y = model.addVars(Q, K + 1, J, vtype=GRB.CONTINUOUS, lb=0, name="y")

        # Objective
        model.setObjective(gp.quicksum(lambda_[q][j] * 
                            (c_fixed[k][j] * y[q, k, j] + c_unit[k][j] * gp.quicksum(u[q, k, i, j] for i in range(I) if i in order_types[q])) 
                            for q in range(Q) for j in range(J) for k in range(K + 1)), GRB.MINIMIZE)

        # Constraints
        for q in range(Q):
            for j in range(J):
                for i in order_types[q]:
                    model.addConstr(gp.quicksum(u[q, k, i, j] for k in range(K + 1)) == 1, f"order{q}_item{i}_city{j}_fulfilled")
                    for k in range(K + 1):
                        model.addConstr(y[q, k, j] >= u[q, k, i, j])

        for i in range(I):
            for k in range(K):  # no inventory constraints for K+1
                model.addConstr(gp.quicksum(gp.quicksum(lambda_[q][j] * u[q, k, i, j] for q in range(Q) if i in order_types[q]) for j in range(J)) <= S[k][i], f"InventoryConstraint_item{i}_facility{k}")

        end_time_init = time.time()
        initialization_duration = end_time_init - start_time_init

        # Turn off the Gurobi output
        model.setParam("OutputFlag", 0)

        # Start timing for optimization
        start_time_opt = time.time()
        
        # Solve model
        model.optimize()
        
        # End timing for optimization
        end_time_opt = time.time()
        optimization_duration = end_time_opt - start_time_opt

        num_vars = model.NumVars
        num_constrs = model.NumConstrs
        optimal_value = None
        u_values = {}
        y_values = {}

        if model.status == GRB.OPTIMAL:
            optimal_value = model.getObjective().getValue()
            # Retrieve the optimal 'u' and 'y' values
            for var in model.getVars():
                if var.varName.startswith('u'):
                    u_values[var.varName] = var.x
                elif var.varName.startswith('y'):
                    y_values[var.varName] = var.x

        return {
            "optimal_u": u_values, 
            "optimal_y": y_values, 
            "num_vars": num_vars, 
            "num_constrs": num_constrs, 
            "optimal_value": optimal_value,
            "initialization_duration": initialization_duration,
            "optimization_duration": optimization_duration
        }

    # Utility functions
    def _get_dimensions(self):
        return len(self.order_fulfillment.df_orders), len(self.order_fulfillment.facility_indices), self.order_fulfillment.num_items, len(self.order_fulfillment.city_indices)

    def _get_costs(self):
        return self.order_fulfillment.fixed_costs, self.order_fulfillment.unit_costs

    def _get_order_types_probs_and_safety_stocks(self):
        order_types = self.order_fulfillment.order_types
        lambda_ = self._reshape_probs(self.order_fulfillment.agg_adjusted_demand_distribution_by_type_by_location)
        S = self.order_fulfillment.safety_stock
        return order_types, lambda_, S

    def _reshape_probs(self, nested_distribution):
        reshape_arrival_prob = []
        for i in nested_distribution:
            for k in i:
                reshape_arrival_prob.append([j for j in k])
        return reshape_arrival_prob

    
    