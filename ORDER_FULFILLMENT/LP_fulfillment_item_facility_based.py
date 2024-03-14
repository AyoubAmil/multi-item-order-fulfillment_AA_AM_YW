## Solving the item-facility based LP formulation from Will Ma

import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB, Model

class ItemFacilityLPSolver:
    def __init__(self, order_fulfillment):
        self.order_fulfillment = order_fulfillment
        self.model = Model()
        self._setup_variables()
        self._setup_objective()
        self._add_constraints()

    def _setup_variables(self):
        Q, K, I, J = self._get_dimensions()
        self.u = self.model.addVars(Q, K + 1, I, J, vtype=GRB.CONTINUOUS, lb=0, name="u")
        self.y = self.model.addVars(Q, K + 1, J, vtype=GRB.CONTINUOUS, lb=0, name="y")

    def _setup_objective(self):
        Q, K, I, J = self._get_dimensions()
        c_fixed, c_unit = self._get_costs()
        order_types, lambda_, S = self._get_order_types_probs_and_safety_stocks()
        
        self.model.setObjective(gp.quicksum(lambda_[q][j] * 
                            (c_fixed[k][j] * self.y[q, k, j] + c_unit[k][j] * gp.quicksum(self.u[q, k, i, j] for i in range(I) if i in order_types[q])) 
                            for q in range(Q) for j in range(J) for k in range(K + 1)), GRB.MINIMIZE)

    def _add_constraints(self):
        Q, K, I, J = self._get_dimensions()
        order_types, lambda_, S = self._get_order_types_probs_and_safety_stocks()

        for q in range(Q):
            for j in range(J):
                for i in order_types[q]:
                    self.model.addConstr(gp.quicksum(self.u[q, k, i, j] for k in range(K + 1)) == 1, f"order{q}_item{i}_city{j}_fulfilled")
                    for k in range(K + 1):
                        self.model.addConstr(self.y[q, k, j] >= self.u[q, k, i, j])

        for i in range(I):
            for k in range(K): # no inventory constraints for K+1
                self.model.addConstr(gp.quicksum(gp.quicksum(lambda_[q][j] * self.u[q, k, i, j] for q in range(Q) if i in order_types[q]) for j in range(J)) <= S[k][i], f"InventoryConstraint_item{i}_facility{k}")

    def optimize(self):
        
        start_time = time.time()
        # Turn off the Gurobi output
        self.model.setParam("OutputFlag", 0)
        
        self.model.optimize()
        
        end_time = time.time()
        optimization_duration = end_time - start_time
        
        num_vars = self.model.NumVars
        num_constrs = self.model.NumConstrs
        optimal_value = None
        u_values = {}
        y_values = {}

        if self.model.status == GRB.OPTIMAL:
            # print('Optimal solution found.')
            optimal_value = self.model.getObjective().getValue()
            
            # Retrieve the optimal 'u' and 'y' values
            for var in self.model.getVars():
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
            "optimization_duration": optimization_duration
        }

    def _get_dimensions(self):
        return len(self.order_fulfillment.df_orders), len(self.order_fulfillment.facility_indices), self.order_fulfillment.num_items, len(self.order_fulfillment.city_indices)

    def _get_costs(self):
        c_fixed = self.order_fulfillment.fixed_costs
        c_unit = self.order_fulfillment.unit_costs
        return c_fixed, c_unit

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
    
    