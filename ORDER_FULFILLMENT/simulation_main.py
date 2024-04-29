import os
from helper_simulation import evaluate_policies

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

evaluate_policies(n_max = 1, T=1, alpha = 0.5, instance = 1, num_order_sequences=num_order_sequences, num_items=num_items, n_0=n_0, p_stock=p_stock, conservative_prob_sequence=conservative_prob_sequence)

# evaluate_policies(n_max, T, alpha, instance, num_order_sequences=num_order_sequences, num_items=num_items, n_0=n_0, p_stock=p_stock, conservative_prob_sequence=conservative_prob_sequence)