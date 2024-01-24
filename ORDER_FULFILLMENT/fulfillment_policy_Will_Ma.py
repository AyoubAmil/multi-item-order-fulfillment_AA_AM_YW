import numpy as np
import copy

class WillMaFulfillmentPolicy:
    def __init__(self, order_fulfillment, optimal_u):
        self.order_fulfillment = order_fulfillment
        self.optimal_u = optimal_u
        self.S = order_fulfillment.safety_stock
        self.c_unit = order_fulfillment.unit_costs
        self.c_fixed = order_fulfillment.fixed_costs
        self.order_types = order_fulfillment.order_types
        self.K = len(order_fulfillment.facility_indices)
        self.XMatrices = self.calculate_XMatrices()

    def calculate_XMatrices(self):
        
        optimal_u = self.optimal_u

        Q = len(self.order_fulfillment.df_orders)
        J = len(self.order_fulfillment.city_indices)
        K = len(self.order_fulfillment.facility_indices)
        I = self.order_fulfillment.num_items
        XMatrices = np.zeros((Q, J, K, I))

        for var_name, value in optimal_u.items():
            # Ensure the variable name follows the expected format
            if var_name.startswith('u['):
                # Extract indices from the variable name
                # Expected format: u[q,k,i,j]
                try:
                    indices = var_name[2:-1].split(',')  # Remove 'u[' and ']', then split by ','
                    q, k, i, j = map(int, indices)
                    if k < K:
                        XMatrices[q, j, k, i] = value
                except ValueError:
                    print(f"Invalid format for variable name: {var_name}")

        return XMatrices


    @staticmethod
    def dilate_round(x, q, j, seed_value):
        np.random.seed(seed_value)
        K = x.shape[0]  # number of facilities
        n = x.shape[1]  # number of items
        fc_tried = np.zeros(n, dtype=int)
        opening_times = np.random.exponential(size=K)  # Exponential distribution
        for i in range(n):
            mask = x[:, i] == 0
            observed_opening = np.full(K, np.inf)
            observed_opening[~mask] = opening_times[~mask] / x[~mask, i]
            fc_tried[i] = np.argmin(observed_opening)
        return fc_tried

    def run(self, arrivals, seed_value):
        tot_cost = 0
        rem_inv = copy.deepcopy(self.S)
        
        # costs = []
        
        for t in range(len(arrivals)):
            
            # cost = 0
            
            order, j = arrivals[t] # order is a tuple of items, j is the location
            q = self.order_types.index(order) # q is now the order index
            if order == ():
                continue

            fc_tried = self.dilate_round(self.XMatrices[q, j], q, j, seed_value)
            fc_used = [False] * (self.K + 1)

            for i in order:
                k = fc_tried[i]
                if k < self.K and rem_inv[k][i] > 0:
                    rem_inv[k][i] -= 1
                else:
                    k = self.K
                fc_used[k] = True
                tot_cost += self.c_unit[k][j]
                # cost += self.c_unit[k][j]

            for k in range(self.K + 1):
                if fc_used[k]:
                    tot_cost += self.c_fixed[k][j]
                    # cost += self.c_fixed[k][j]
        #     costs.append(cost)
        # print(costs)
        return tot_cost



