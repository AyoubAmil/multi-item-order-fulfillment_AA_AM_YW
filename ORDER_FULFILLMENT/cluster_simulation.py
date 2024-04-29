# cluster_simulation.py

import pickle
import numpy as np
import os
import csv
from scipy import stats
from order_fulfillment_environment_notidentical_arrival_probs import OrderFulfillment
from cluster_fulfillment_policy_notidentical_arrival_probs import FulfillmentPolicy
from fulfillment_policy_Will_Ma import WillMaFulfillmentPolicy

num_items = 20
n_0 = 5
p_stock = 0.75
CSL=0.5

conservative_prob_sequence=[0, 0.01, 0.05, 0.1, 0.15, 0.2, 1]

num_order_sequences = 100

facility_path = "fulfillment_centers_warmup.csv"
cities_path = "cities_warmup.csv"
home_path = "/hpc/home/aa554/Data/"
facilities_path = os.path.join(home_path, facility_path)
cities_path =  os.path.join(home_path, cities_path)


def DLP_ours_results(fulfillment_policy, print_optimal_values=False):
    # Call the method and get the results
    results = fulfillment_policy.get_optimization_results

    # Check if the optimization was successful and optionally print the results
    if results is not None:
        if print_optimal_values:
            print("Optimal 'x' values:")
            for q, x_values in results["x_values"].items():
                print(f"Order Type {q}:")
                for method_index, value in x_values.items():
                    print(f"  Method {method_index}: {value}")

            print(f"\nNumber of decision variables: {results['num_vars']}")
            print(f"Number of constraints: {results['num_constrs']}")
            print(f"Optimal objective value: {results['optimal_value']}")

        return results["num_vars"], results["num_constrs"], results["optimal_value"]
    
    else:
        print("Optimization did not find an optimal solution for our formulation.")
        return None


def evaluate_policies(n_max, T, alpha, instance, num_order_sequences=num_order_sequences, num_items=num_items, n_0=n_0, p_stock=p_stock, conservative_prob_sequence=conservative_prob_sequence):
    
    dir_results = f'/hpc/home/aa554/LP_results_instance={instance}/n_max={n_max}/T={T}_alpha={alpha}'

    # Load LP results from files (make sure to run cluster_solve_LP.py first with same order_fulfillment parameters)
    with open(os.path.join(dir_results, 'LP_solution.pkl'), 'rb') as f:
        LP_solution = pickle.load(f)
    with open(os.path.join(dir_results, 'methods.pkl'), 'rb') as f:
        methods = pickle.load(f)
    with open(os.path.join(dir_results, 'sizes.pkl'), 'rb') as f:
        sizes = pickle.load(f)
    with open(os.path.join(dir_results, 'consumption_probability_lists.pkl'), 'rb') as f:
        consumption_probability_lists = pickle.load(f)
    with open(os.path.join(dir_results, 'get_optimization_results.pkl'), 'rb') as f:
        get_optimization_results = pickle.load(f)
    
    dir_results_wm = f'/hpc/home/aa554/LP_results_WM_instance={instance}/n_max={n_max}/T={T}_alpha={alpha}'
        
    with open(os.path.join(dir_results_wm, 'wm_num_vars.pkl'), 'rb') as f:
        wm_num_vars = pickle.load(f)
    with open(os.path.join(dir_results_wm, 'wm_num_constrs.pkl'), 'rb') as f:
        wm_num_constrs = pickle.load(f)
    with open(os.path.join(dir_results_wm, 'wm_optimal_value.pkl'), 'rb') as f:
        wm_optimal_value = pickle.load(f)
    with open(os.path.join(dir_results_wm, 'optimal_u.pkl'), 'rb') as f:
        optimal_u = pickle.load(f)
    with open(os.path.join(dir_results_wm, 'optimal_y.pkl'), 'rb') as f:
        optimal_y = pickle.load(f)
    with open(os.path.join(dir_results_wm, 'wm_optimization_duration.pkl'), 'rb') as f:
        wm_optimization_duration = pickle.load(f)
    with open(os.path.join(dir_results_wm, 'our_optimization_duration.pkl'), 'rb') as f:
        our_optimization_duration = pickle.load(f)
    
    # This object has the be the same as the one used to generate the LP results in the cluster_solve_LP.py    
    order_fulfillment = OrderFulfillment(num_items=num_items, n_max=n_max, n_0=n_0, p_stock=p_stock, T=T, CSL=CSL, 
                                        facilities_data=facilities_path, 
                                        cities_data=cities_path, 
                                        prob_seed_value=instance, 
                                        order_seed_value=instance, 
                                        inv_seed_value=instance, 
                                        alpha=alpha)

    # Now create an instance of FulfillmentPolicy with these parameters
    fulfillment_policy = FulfillmentPolicy(order_fulfillment, LP_solution, methods, sizes, consumption_probability_lists, get_optimization_results)
    
    our_num_vars, our_num_constrs, our_optimal_value = DLP_ours_results(fulfillment_policy)
    
    # Instantiate WillMaFulfillmentPolicy
    will_ma_policy = WillMaFulfillmentPolicy(order_fulfillment, optimal_u)
    
    results_over_conservative_probs = {}
    
    for conservative_prob in conservative_prob_sequence:
        # Generate magician problems (dictionary where the keys are (i,k))
        cdfs_magician_problems = fulfillment_policy.solve_cdfs_magician_problems(conservative_prob=conservative_prob)
        
        # FOR EACH INSTANCE, GENERATE DIFFERENT ORDER SEQUENCES (i.e. order arrivals through time)
        order_sequences = np.arange(1, num_order_sequences + 1) # seed_value for each order sequence

        # Initialize counters for number of times each policy is better
        count_fulfillment_policy = 0
        count_will_ma_policy = 0
        tie = 0

        # Lists to store costs
        fulfillment_costs_list = []
        always_accept_costs_list = []
        will_ma_costs_list = []
        
        # Lists to store sampled orders, samples methods, and accepts decisions
        our_policy_orders_methods_decisions = {}
        aa_policy_orders_methods_decisions = {}
        
        for order_sequence in order_sequences:
            
            magician_problems = fulfillment_policy.solve_magician_problems(conservative_prob=conservative_prob, cdfs_magician_problems=cdfs_magician_problems)
            
            # Initialize inventory consumption for our fulfillment policy
            inventory_consumption = fulfillment_policy.initialize_inventory_consumption()
            sampled_orders_index, sampled_orders, sampled_methods, accepts_decisions, fulfillment_costs = fulfillment_policy.new_fulfillment_policy_end_modified(inventory_consumption, magician_problems, seed_value=order_sequence)
            fulfillment_policy.check_consistency(inventory_consumption)
            total_fulfillment_cost = sum(fulfillment_costs)
            our_policy_orders_methods_decisions["orders"] = sampled_orders
            our_policy_orders_methods_decisions["methods"] = sampled_methods
            our_policy_orders_methods_decisions["accepts"] = accepts_decisions
        
            # Initialize inventory consumption for always_accept_policy
            inventory_consumption_aa = fulfillment_policy.initialize_inventory_consumption()
            sampled_orders_index_aa, sampled_orders_aa, sampled_methods_aa, accepts_decisions_aa, fulfillment_costs_aa = fulfillment_policy.always_accept_policy(inventory_consumption_aa, seed_value=order_sequence)
            fulfillment_policy.check_consistency(inventory_consumption)
            total_always_accept_cost = sum(fulfillment_costs_aa)
            aa_policy_orders_methods_decisions["orders"] = sampled_orders_aa
            aa_policy_orders_methods_decisions["methods"] = sampled_methods_aa
            aa_policy_orders_methods_decisions["accepts"] = accepts_decisions_aa

            # Will Ma's policy
            will_ma_cost = will_ma_policy.run(sampled_orders, seed_value=order_sequence)

            # Append cost of order sequence to lists
            fulfillment_costs_list.append(total_fulfillment_cost)
            always_accept_costs_list.append(total_always_accept_cost)
            will_ma_costs_list.append(will_ma_cost)

            # Check which policy has lower costs (for given order arrivals) and update counters accordingly
            if total_fulfillment_cost < will_ma_cost:
                count_fulfillment_policy += 1
            elif will_ma_cost < total_fulfillment_cost:
                count_will_ma_policy += 1
            elif total_fulfillment_cost == will_ma_cost:
                tie += 1

        # 95% CONFIDENCE INTERVAL for the difference (for fixed instance) between our policy and Will Ma's policy
        cost_differences = np.array(fulfillment_costs_list) - np.array(will_ma_costs_list)
        mean_difference = np.mean(cost_differences)
        std_deviation = np.std(cost_differences, ddof=1)
        n = len(cost_differences)
        standard_error = std_deviation / np.sqrt(n) # Calculate the standard error and the t-score for 95% confidence
        t_score = stats.t.ppf(0.975, df=n-1)  # two-tailed 95% confidence, so 0.975
        confidence_interval = (mean_difference - t_score * standard_error, mean_difference + t_score * standard_error)
        
        # 95% CONFIDENCE INTERVAL for the difference between our policy and AA policy
        cost_differences_aa = np.array(fulfillment_costs_list) - np.array(always_accept_costs_list)
        mean_difference_aa = np.mean(cost_differences_aa)
        std_deviation_aa = np.std(cost_differences_aa, ddof=1)
        n_aa = len(cost_differences_aa)
        standard_error_aa = std_deviation_aa / np.sqrt(n_aa)  # Calculate the standard error
        t_score_aa = stats.t.ppf(0.975, df=n_aa-1)  # two-tailed 95% confidence, so 0.975
        confidence_interval_aa = (mean_difference_aa - t_score_aa * standard_error_aa, mean_difference_aa + t_score_aa * standard_error_aa)
        
        expected_policy_cost = round(np.mean(fulfillment_costs_list),2) # expected cost over the number of order sequences
        expected_policy_cost_aa = round(np.mean(always_accept_costs_list),2) # expected cost over the number of order sequences
        expected_cost_will_ma = round(np.mean(will_ma_costs_list), 2)
        expected_cost_difference = round(expected_policy_cost - expected_cost_will_ma,2)
        expected_cost_difference_aa = round(expected_policy_cost - expected_policy_cost_aa,2)  # expected cost difference between our policy and AA policy
        
        percent_ours_better_over_order_arrivals = count_fulfillment_policy / num_order_sequences * 100
        percent_equal_over_order_arrivals = tie / num_order_sequences * 100
        percent_wm_better_over_order_arrivals = count_will_ma_policy / num_order_sequences * 100
    
        results_per_conservative_probs = {
            "fulfillments_cost_our_policy_per_order_sequence": fulfillment_costs_list,
            "fulfillments_cost_aa_per_order_sequence": always_accept_costs_list,
            "fulfillments_cost_will_ma_per_order_sequence": will_ma_costs_list,
            "expected_cost_our_policy_over_order_sequences": expected_policy_cost,
            "expected_cost_aa_over_order_sequence": expected_policy_cost_aa,
            "expected_cost_will_ma_over_order_sequences": expected_cost_will_ma,
            "expected_cost_difference_over_order_sequence_our_cost-wm_cost": expected_cost_difference,
            "95%_CI_cost_difference_our_cost-wm_cost": confidence_interval,
            "expected_cost_difference_over_order_sequence_our_cost-aa_cost": expected_cost_difference_aa,
            "95%_CI_cost_difference_our_cost-aa_cost": confidence_interval_aa,
            "percent_ours_better_over_order_arrivals": percent_ours_better_over_order_arrivals,
            "percent_equal_over_order_arrivals": percent_equal_over_order_arrivals,
            "percent_wm_better_over_order_arrivals": percent_wm_better_over_order_arrivals,
            "(our_num_vars, our_num_constrs)": (our_num_vars, our_num_constrs),
            "our_optimal_value": our_optimal_value,
            "our_optimization_duration": our_optimization_duration,
            "(wm_num_vars, wm_num_constrs)": (wm_num_vars, wm_num_constrs),
            "wm_optimal_value": wm_optimal_value,
            "wm_optimization_duration": wm_optimization_duration,
        }
    
        results_over_conservative_probs[f"{conservative_prob}"] = results_per_conservative_probs
    
    res_path = f"/hpc/home/aa554/Results/num_items={num_items}_n_max={n_max}_n_0={n_0}/instance={instance}_T={T}_alpha={alpha}"
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    # Define the CSV file path
    csv_file_path = os.path.join(res_path, f'instance={instance}_T={T}_alpha={alpha}.csv')

    # Prepare data for CSV
    data_for_csv = []
    for prob, results in results_over_conservative_probs.items():
        for key, value in results.items():
            if isinstance(value, list):
                for idx, val in enumerate(value):
                    data_row = {
                        "Conservative Probability": prob,
                        "Metric": key,
                        "Order Sequence": idx + 1,
                        "Value": val
                    }
                    data_for_csv.append(data_row)
            elif isinstance(value, tuple):
                data_row = {
                    "Conservative Probability": prob,
                    "Metric": key,
                    "Order Sequence": "N/A",
                    "Value": f"({value[0]:.2f}, {value[1]:.2f})"
                }
                data_for_csv.append(data_row)
            else:
                data_row = {
                    "Conservative Probability": prob,
                    "Metric": key,
                    "Order Sequence": "N/A",
                    "Value": value
                }
                data_for_csv.append(data_row)

    # Write to CSV
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ["Conservative Probability", "Metric", "Order Sequence", "Value"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data_for_csv:
            writer.writerow(row)
            
# evaluate_policies(n_max, T, alpha, instance, num_order_sequences=num_order_sequences, num_items=num_items, n_0=n_0, p_stock=p_stock, conservative_prob_sequence=conservative_prob_sequence)