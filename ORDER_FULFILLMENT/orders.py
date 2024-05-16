# Create dataframe of order types

import pandas as pd

def create_orders_df(order_types):
    temp_lista = []
    for i in range(len(order_types)):
        temp_lista.append(order_types[i])
    df_orders = pd.DataFrame()
    df_orders['order'] = temp_lista
    return df_orders

# Create dataframe of order types and locations
def create_orders_location_df(order_types, num_cities):
    temp_lista_location = []
    for i in range(len(order_types)):
        for j in range(num_cities):
            temp_lista_location.append((order_types[i],j))
    df_orders_location = pd.DataFrame()
    df_orders_location['(order,location)'] = temp_lista_location
    return df_orders_location