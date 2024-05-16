 # Utility functions
 
import random
from itertools import chain, combinations
from math import *

# Calculate the great circle distance in miles between two points on the Earth
def haversine(lon1, lat1, lon2, lat2, inv):
        """
        Calculate the great circle distance in miles between two points on the Earth
        (specified in decimal degrees), given their inventory status.
        """
        if inv == 0: 
            return 10**6 # Return a large distance value if the facility has no inventory
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 3959 # Radius of Earth in miles
        return c * r
 
def random_combination(iterable, r):
    """Random selection of r elements from iterable"""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

def powerset(iterable):
        """Return the powerset of the given iterable."""
        s = list(iterable)
        # Generate a powerset (all possible subsets) of the input iterable by iterating through all possible combinations
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    
# Reshape a nested (3D) distribution
def reshape_probs(nested_distribution):
    """reshape a nested (3D) distribution"""
    reshape_arrival_prob = []
    for i in nested_distribution:
        for k in i:
            for j in k:
                reshape_arrival_prob.append(j)
    return reshape_arrival_prob
    
# Reshape a nested (4D) distribution
def reshape_modified_probs(nested_distribution):
    reshape_arrival_prob = []
    for t in range(len(nested_distribution)):
        reshape_arrival_prob.append(reshape_probs(nested_distribution[t]))
    return reshape_arrival_prob
