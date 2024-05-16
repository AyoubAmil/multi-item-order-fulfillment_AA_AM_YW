import os
import numpy as np
import pickle
from order_fulfillment_network import OrderFulfillment
from LP_fulfillment_notidentical_arrival_probs import SolvingLP
from LP_fulfillment_item_facility_based import ItemFacilityLPSolver

# Network with 10 cities and 5 facilities
facility_path = "fulfillment_centers_warmup.csv"
cities_path = "cities_warmup.csv"

# Network with 99 cities and 10 facilities
# facility_path = "fulfillment_centers.csv"
# cities_path = "cities.csv"

home_path = os.getcwd() + "/ORDER_FULFILLMENT/Data" # Change this to the path where the data is stored
facilities_path = os.path.join(home_path, facility_path)
cities_path =  os.path.join(home_path, cities_path)

num_instances = 30
instances = np.arange(1, num_instances + 1)
T_values = [10**3]
alpha_values = [0.5]

num_items = 20
n_max = 2
# n_max = 5 Uncomment this line for larger order sizes
n_0 = 5
p_stock = 0.75

def DLP_WillMa(order_fulfillment, print_optimal_values=False):
    
    solver = ItemFacilityLPSolver(order_fulfillment)

    results = solver.optimize()

    # Extract the results
    optimal_u = results["optimal_u"]
    optimal_y = results["optimal_y"]
    num_vars = results["num_vars"]
    num_constrs = results["num_constrs"]
    optimal_value = results["optimal_value"]
    initialization_duration = results["initialization_duration"]
    optimization_duration = results["optimization_duration"]

    # Check if optimization was successful and print results
    if optimal_u and optimal_y:
        
        if print_optimal_values:
            print("Optimal 'u' values:")
            for var_name, value in optimal_u.items():
                print(f"{var_name} = {value}")

            print("\nOptimal 'y' values:")
            for var_name, value in optimal_y.items():
                print(f"{var_name} = {value}")

        return num_vars, num_constrs, optimal_value, optimal_u, optimal_y, initialization_duration, optimization_duration
    
    else:
        print("Optimization did not find an optimal solution for Will Ma's formulation.")


for instance in instances:
    for T in T_values:
        for alpha in alpha_values:
            # Solve our LP for the base value of T
            order_fulfillment = OrderFulfillment(num_items=num_items, n_max=n_max, n_0=n_0, p_stock=p_stock, T=T, 
                                                        facilities_data=facilities_path, 
                                                        cities_data=cities_path, 
                                                        prob_seed_value=instance, 
                                                        order_seed_value=instance, 
                                                        inv_seed_value=instance, 
                                                        alpha=alpha)

            solving_LP_instance = SolvingLP(order_fulfillment)
            LP_solution, methods, sizes, num_vars, num_constrs, optimal_value, our_initialization_duration, our_optimization_duration = solving_LP_instance.optimize_LP_relaxation()
        
            wm_num_vars, wm_num_constrs, wm_optimal_value, optimal_u, optimal_y, wm_initialization_duration, wm_optimization_duration = DLP_WillMa(order_fulfillment)
            
            # SAVE WM RESULTS
            instance_dir_wm = f'LP_results_WM_instance={instance}/n_max={n_max}/T={T}_alpha={alpha}'
            
            if not os.path.exists(instance_dir_wm):
                os.makedirs(instance_dir_wm)

            with open(f'{instance_dir_wm}/wm_num_vars.pkl', 'wb') as f:
                pickle.dump(wm_num_vars, f)
            with open(f'{instance_dir_wm}/wm_num_constrs.pkl', 'wb') as f:
                pickle.dump(wm_num_constrs, f)
            with open(f'{instance_dir_wm}/wm_optimal_value.pkl', 'wb') as f:
                pickle.dump(wm_optimal_value, f)
            with open(f'{instance_dir_wm}/optimal_u.pkl', 'wb') as f:
                pickle.dump(optimal_u, f)
            with open(f'{instance_dir_wm}/optimal_y.pkl', 'wb') as f:
                pickle.dump(optimal_y, f)
            with open(f'{instance_dir_wm}/wm_initialization_duration.pkl', 'wb') as f:
                    pickle.dump(wm_initialization_duration, f)
            with open(f'{instance_dir_wm}/wm_optimization_duration.pkl', 'wb') as f:
                pickle.dump(wm_optimization_duration, f)

            # SAVE OUR RESULTS
            consumption_probability_lists = solving_LP_instance.calculate_probabilities_of_consumption(LP_solution)
            
            get_optimization_results = {"x_values": LP_solution, "num_vars": num_vars, 
                "num_constrs": num_constrs, "optimal_value": optimal_value}

            # Save our results
            instance_dir = f'LP_results_instance={instance}/n_max={n_max}/T={T}_alpha={alpha}'
            
            if not os.path.exists(instance_dir):
                os.makedirs(instance_dir)

            with open(f'{instance_dir}/LP_solution.pkl', 'wb') as f:
                pickle.dump(LP_solution, f)
            with open(f'{instance_dir}/methods.pkl', 'wb') as f:
                pickle.dump(methods, f)
            with open(f'{instance_dir}/sizes.pkl', 'wb') as f:
                pickle.dump(sizes, f)
            with open(f'{instance_dir}/consumption_probability_lists.pkl', 'wb') as f:
                pickle.dump(consumption_probability_lists, f)
            with open(f'{instance_dir}/get_optimization_results.pkl', 'wb') as f:
                pickle.dump(get_optimization_results, f)
            with open(f'{instance_dir}/our_initialization_duration.pkl', 'wb') as f:
                    pickle.dump(our_initialization_duration, f)
            with open(f'{instance_dir}/our_optimization_duration.pkl', 'wb') as f:
                    pickle.dump(our_optimization_duration, f)

