###########################################################################################################
# Genetic Algorithm for solving Travelling Salesman Problem - main
# Authors: Julian Gruber, Nikolaus Lajtai
###########################################################################################################

import pandas as pd
import folium
from tspgenetic import Population

# Constants for loading the cities
MAX_NR_CITIES = 100
MIN_NR_RESIDENTS = 50000
# COUNTRY_LIST = ['Germany', 'Austria', 'France', 'Hungary', 'Italia']
COUNTRY_LIST = ['Germany','Austria', 'France', 'Hungary', 'Italia', 'United Kingdom']

# Constants for genetic algorithm
MAX_GENERATIONS = 600
NUMBER_INDIVIDUALS = 2000
START_CITY = "Graz"

###########################################################################################################
# Function Init
###########################################################################################################

# Filter Cities by country name, min city size and my num of cities
def filter_cities(dataframe, country_list, nr_cities=100, min_city_size = 50000):          
    dataframe = dataframe[dataframe["country"].isin(country_list)]
    dataframe = dataframe[dataframe["population"] > min_city_size]
    dataframe = dataframe.drop(['city_ascii', 'iso2', 'iso3', 'admin_name', 'population','capital', 'id'], axis=1)
    dataframe = dataframe[0:nr_cities]
    dataframe = dataframe.reset_index(drop=True)
    
    return dataframe

# Plotting cities on a map
def plot_markers_and_connection(pd_dataframe: pd.DataFrame, path, zoom_factor: int = 5):
    # mean lat and lng for map center
    lat_mean = pd_dataframe['lat'].mean()
    lng_mean = pd_dataframe['lng'].mean()
    city_map = folium.Map(location=[lat_mean, lng_mean], zoom_start=zoom_factor)

    # look up the indices of the cities in path
    sort_list = list()
    for city in path:
        city_index = pd_dataframe.index[pd_dataframe['city']==city].tolist()[0]
        sort_list.append(city_index)

    # placing connection lines on map
    coord_list = [[pd_dataframe['lat'][i], pd_dataframe['lng'][i]] for i in sort_list]
    folium.PolyLine(coord_list, color='red').add_to(city_map)

    # adding marker on map with citie names
    list_coor=pd_dataframe[['city','lat','lng']].values.tolist()

    for i in list_coor:
        city_map.add_child(folium.Marker(location=[i[1],i[2]], popup=i[0],icon=folium.Icon(color='green')))

    # saving map
    city_map.save('assign_1_city_map.html')


###########################################################################################################
# Execution
###########################################################################################################
if __name__ == "__main__":
    # read in data from csv
    df=pd.read_csv("worldcities.csv")

    filtered_df = filter_cities(df, COUNTRY_LIST, MAX_NR_CITIES, MIN_NR_RESIDENTS)
    print(filtered_df)

    # init population of NUMBER_INDIVIDUALS random permutations/travel routes
    pop = Population(filtered_df, START_CITY, NUMBER_INDIVIDUALS)
    
    best = pop.return_best() # best of all generations

    # reproduce for a given number of generations
    for gen in range(1,MAX_GENERATIONS+1):

        # Each time reproduce() is called, all individuals produce a mutated offspring. 
        # A mutation means that two cities swap places in the travel route. 
        # Then either the parent or the offspring is kept, based on the pathlength.
        pop.reproduce()

        # Store the individual with the shortest pahtlength
        tmp_best = pop.return_best()
        
        # Update best of all generations
        if (tmp_best.pathlength < best.pathlength):
            best = tmp_best
    
    # Print the individual with the shortest pathlength and the total pathlength
        if gen%10 == 0:
            print("\nGen {}:\nPath: {}\nDistance:{}".format(gen, tmp_best, tmp_best.pathlength))

    print("\n---Best---\nPath: {}\nDistance:{}".format(gen, tmp_best, tmp_best.pathlength))
    # plot everything on the map
    plot_markers_and_connection(filtered_df, best.state, zoom_factor=5)
