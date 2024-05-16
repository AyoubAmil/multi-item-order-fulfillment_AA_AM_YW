# Data preparation for order fulfillment

import pandas as pd

# Load facility data
def load_fulfillment_centers(file_path, use_single_facility=False):
    """Loads the fulfillment center data from a CSV file."""
    fulfillment_centers = pd.read_csv(file_path)
    if use_single_facility:
        fulfillment_centers = fulfillment_centers.iloc[:1]
    return fulfillment_centers
    
# Load data about cities
def load_cities(file_path, use_single_city=False):
    """Loads the city data from a CSV file."""
    cities_df = pd.read_csv(file_path)
    if use_single_city:
        cities_df = cities_df.iloc[:1]
    return cities_df

# Items
def generate_items_list(num_items):
    """Generates a list of item indices."""
    return list(range(num_items))

# Generate facility indices and locations (longitude (y) and latitude (x))
def generate_facility_indices_and_locations(fulfillment_centers_df):
    """Generates the indices and locations of fulfillment centers."""
    num_facilities = fulfillment_centers_df.shape[0]
    facility_indices = list(range(num_facilities))
    facility_locations = [(fulfillment_centers_df["Longitude"][r], fulfillment_centers_df["Latitude"][r]) for r in range(num_facilities)]
    return facility_indices, facility_locations

# Generate city indices and locations (longitude (y) and latitude (x))
def generate_city_indices_and_locations(cities_df):
    """Generates the indices and locations of cities."""
    num_cities = cities_df.shape[0]
    city_indices = list(range(num_cities))
    city_locations = [(cities_df["Longitude"][r], cities_df["Latitude"][r]) for r in range(num_cities)]
    return city_indices, city_locations

# Get population data and total population
def get_population_data(cities_df):
    population = cities_df["Population"]
    total_population = sum(population)
    return population, total_population

