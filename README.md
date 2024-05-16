
# Multi-Item Order Fulfillment Revisited: LP Formulation and Prophet Inequality

This repository contains the data and simulation code for the paper: "Multi-Item Order Fulfillment Revisited: LP
Formulation and Prophet Inequality", Ayoub Amil, Ali Makhdoumi, Yehua Wei. The codebase is designed following the model from Section 2 of the paper. The simulation is described in detail in Section 7 and Appendix C.

## Modules

The simulation is divided into 8 main modules, each of which serves a specific purpose. The main modules are as follows:

1. `order_fulfillment_network.py`: Implements the multi-item order fulfillment model from Section 2 of the paper. In order to implement this, several modules are defined:
    - `demand_distribution.py`: Defines the demand distribution for orders.
    - `methods.py`: Defines methods for order types.
    - `orders.py`: Creates dataframes of orders.
    - `costs.py`: Calculates various costs associated with fulfillment.
    - `data_preparation.py`: Prepares data for simulation.
    - `utilities.py`: Provides utility functions for various tasks.
    - `inventory.py`: Defines inventory levels.

2. `cluster_fulfillment_policy_notidentical_arrival_probs.py`: Implements the fulfillment policy from the paper and the always accept policy.
3. `cluster_simulation.py`: Runs the simulation and saves the results. 
4. `cluster_solve_LP.py`: Solves the required LPs.
5. `fulfillment_policy_Will_Ma.py`: Implements Will Ma's fulfillment policy based on Ma (2023).
6. `LP_fulfillment_notidentical_arrival_probs.py`: Defines the method-based LP formulation.
7. `LP_fulfillment_item_facility_based.py`: Defines the item-facility-based LP formulation.
8. `magician_problem.py`: Implements the Magician problem.

## Data

The `Data` directory contains the necessary datasets for running the simulations. Ensure that the data files are correctly placed in this directory before running the scripts. Moreover, ensure that the paths in the scripts are correctly defined to read the data files.

In the `Data` directory you will find the following files:
- `cities_warmup_test.csv`: This dataset contains coordinates of 2 US cities and it is only used for testing purposes.
- `cities_warmup.csv`: This dataset contains coordinates of 10 US cities
- `cities.csv`: This dataset contains coordinates of 99 US cities
- `fulfillment_centers_warmup_test.csv`: This dataset contains coordinates of 2 US fulfillment centers and it is only used for testing purposes.
- `fulfillment_centers_warmup.csv`: This dataset contains coordinates of 5 US fulfillment centers.
- `fulfillment_centers.csv`: This dataset contains coordinates of 10 US fulfillment centers.
