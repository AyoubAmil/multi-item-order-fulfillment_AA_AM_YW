import os
import numpy as np
import pickle
from order_fulfillment_environment_notidentical_arrival_probs import OrderFulfillment
from LP_fulfillment_notidentical_arrival_probs import SolvingLP
from LP_fulfillment_item_facility_based import ItemFacilityLPSolver

facility_path = "fulfillment_centers_warmup.csv"
cities_path = "cities_warmup.csv"
home_path = "/Users/ayoubamil/Documents/GitHub/multi-item-order-fulfillment_AA_AM_YW/ORDER_FULFILLMENT/Data"
facilities_path = os.path.join(home_path, facility_path)
cities_path =  os.path.join(home_path, cities_path)

num_instances = 30
instances = np.arange(1, num_instances + 1)
# T_values = [10**3, 10**4, 10**5, 10**6]
T_values = [10**3]
# alpha_values = [0.1, 0.5, 0.75, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1]
# alpha_values = [0.5, 0.9, 1]
alpha_values = [0, 0.5, 1]
T_base = 10**3
alpha_base = 0.5

num_items = 20
n_max = 2
# n_max = 5
n_0 = 5
p_stock = 0.75
CSL=0.5

def DLP_WillMa(order_fulfillment, print_optimal_values=False):
    
    solver = ItemFacilityLPSolver(order_fulfillment)

    results = solver.optimize()

    # Extract the results
    optimal_u = results["optimal_u"]
    optimal_y = results["optimal_y"]
    num_vars = results["num_vars"]
    num_constrs = results["num_constrs"]
    optimal_value = results["optimal_value"]
    optimization_duration = results["optimization_duration"]

    # Step 4: Check if optimization was successful and print results
    if optimal_u and optimal_y:
        
        if print_optimal_values:
            print("Optimal 'u' values:")
            for var_name, value in optimal_u.items():
                print(f"{var_name} = {value}")

            print("\nOptimal 'y' values:")
            for var_name, value in optimal_y.items():
                print(f"{var_name} = {value}")

        return num_vars, num_constrs, optimal_value, optimal_u, optimal_y, optimization_duration
    
    else:
        print("Optimization did not find an optimal solution for Will Ma's formulation.")


for instance in instances:
    # Solve our LP for the base value of T
    order_fulfillment_base = OrderFulfillment(num_items=num_items, n_max=n_max, n_0=n_0, p_stock=p_stock, T=T_base, CSL=CSL, 
                                                facilities_data=facilities_path, 
                                                cities_data=cities_path, 
                                                prob_seed_value=instance, 
                                                order_seed_value=instance, 
                                                inv_seed_value=instance, 
                                                alpha=alpha_base) # This alpha does not really matter

    solving_LP_instance_base = SolvingLP(order_fulfillment_base)
    LP_solution_base, methods_base, sizes_base, num_vars_base, num_constrs_base, optimal_value_base, our_optimization_duration = solving_LP_instance_base.optimize_LP_relaxation()

    for T in T_values:
        scale_factor = T / T_base
        LP_solution_scaled = {}
        for q, inner_dict in LP_solution_base.items():
            scaled_inner_dict = {m: value * scale_factor for m, value in inner_dict.items()}
            LP_solution_scaled[q] = scaled_inner_dict
        
        # order_fulfillment_wm = OrderFulfillment(num_items=num_items, n_max=n_max, n_0=n_0, p_stock=p_stock, T=T, CSL=CSL, 
        #                                         facilities_data=facilities_path, 
        #                                         cities_data=cities_path, 
        #                                         prob_seed_value=instance, 
        #                                         order_seed_value=instance, 
        #                                         inv_seed_value=instance, 
        #                                         alpha=alpha_base)
        
        # wm_num_vars, wm_num_constrs, wm_optimal_value, optimal_u, optimal_y, wm_optimization_duration = DLP_WillMa(order_fulfillment_wm)
        
        # # Save WM results
        # instance_dir_wm = f'LP_results_WM_instance={instance}/n_max={n_max}/T={T}'
        
        # if not os.path.exists(instance_dir_wm):
        #     os.makedirs(instance_dir_wm)

        # with open(f'{instance_dir_wm}/wm_num_vars.pkl', 'wb') as f:
        #     pickle.dump(wm_num_vars, f)
        # with open(f'{instance_dir_wm}/wm_num_constrs.pkl', 'wb') as f:
        #     pickle.dump(wm_num_constrs, f)
        # with open(f'{instance_dir_wm}/wm_optimal_value.pkl', 'wb') as f:
        #     pickle.dump(wm_optimal_value, f)
        # with open(f'{instance_dir_wm}/optimal_u.pkl', 'wb') as f:
        #     pickle.dump(optimal_u, f)
        # with open(f'{instance_dir_wm}/optimal_y.pkl', 'wb') as f:
        #     pickle.dump(optimal_y, f)
        # with open(f'{instance_dir_wm}/wm_optimization_duration.pkl', 'wb') as f:
        #     pickle.dump(wm_optimization_duration, f)
        # with open(f'{instance_dir_wm}/our_optimization_duration.pkl', 'wb') as f:
        #         pickle.dump(our_optimization_duration, f)
        
        for alpha in alpha_values:
            order_fulfillment = OrderFulfillment(num_items=num_items, n_max=n_max, n_0=n_0, p_stock=p_stock, T=T, CSL=CSL, 
                                                facilities_data=facilities_path, 
                                                cities_data=cities_path, 
                                                prob_seed_value=instance, 
                                                order_seed_value=instance, 
                                                inv_seed_value=instance, 
                                                alpha=alpha)
        
            wm_num_vars, wm_num_constrs, wm_optimal_value, optimal_u, optimal_y, wm_optimization_duration = DLP_WillMa(order_fulfillment)
            
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
            with open(f'{instance_dir_wm}/wm_optimization_duration.pkl', 'wb') as f:
                pickle.dump(wm_optimization_duration, f)
            with open(f'{instance_dir_wm}/our_optimization_duration.pkl', 'wb') as f:
                    pickle.dump(our_optimization_duration, f)

            # SAVE OUR RESULTS
            solving_LP_instance = SolvingLP(order_fulfillment)
            # consumption_probability_lists = solving_LP_instance.calculate_probabilities_of_consumption(LP_solution_scaled, sizes_base)
            consumption_probability_lists = solving_LP_instance.calculate_probabilities_of_consumption(LP_solution_scaled)
            get_optimization_results = {"x_values": LP_solution_scaled, "num_vars": num_vars_base, 
                "num_constrs": num_constrs_base, "optimal_value": optimal_value_base * scale_factor}

            # Save our results
            instance_dir = f'LP_results_instance={instance}/n_max={n_max}/T={T}_alpha={alpha}'
            
            if not os.path.exists(instance_dir):
                os.makedirs(instance_dir)

            with open(f'{instance_dir}/LP_solution.pkl', 'wb') as f:
                pickle.dump(LP_solution_scaled, f)
            with open(f'{instance_dir}/methods.pkl', 'wb') as f:
                pickle.dump(methods_base, f)
            with open(f'{instance_dir}/sizes.pkl', 'wb') as f:
                pickle.dump(sizes_base, f)
            with open(f'{instance_dir}/consumption_probability_lists.pkl', 'wb') as f:
                pickle.dump(consumption_probability_lists, f)
            with open(f'{instance_dir}/get_optimization_results.pkl', 'wb') as f:
                pickle.dump(get_optimization_results, f)

