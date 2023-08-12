###########################################################################################################
# Genetic Algorithm for solving Travelling Salesman Problem - Function Library
# Authors: Julian Gruber, Nikolaus Lajtai
###########################################################################################################

import pandas as pd
import numpy as np
from geopy import distance
import random, copy

class DistanceMatrix():
    """
    Stores the distances between all cities given in the cities_df at 
     initialization

    Attributes:
    ----
    matrix: ndarray
        distances between all cities of cities_df
    index_dict: dict
        indices of all cities to acces distances in matrix

    Methods:
    ---
    compute_marix(cities_df)
        compute the distance matrix
    save_city_indices(cities_df)
        save the index of each city corresponding to the matrix
    get_distance(city1, city2)
        look up distance between the two cities from the matrix
    """
    def __init__(self, cities_df: pd.DataFrame):
        """
        Args:
        ---
        cities_df: DataFrame
            columns -> city: str, lat: float, lng: float
        """
        self.matrix = self.compute_matrix(cities_df)
        self.index_dict = self.save_city_indices(cities_df)

    def compute_matrix(self, cities_df: pd.DataFrame):
        """compute the distances between all cities of the city dataframe

        Args:
        ---
        cities_df: DataFrame
            columns -> city: str, lat: float, lng: float
            
        Retruns:
        ---
        np.ndarray
            symmetric matrix of distances from city in column to city in
             row
        """
        dmat = np.zeros((len(cities_df), len(cities_df)))
        for i, row1 in cities_df.iterrows():
            for j, row2 in cities_df.iterrows():
                if j>i:
                    coordinates1 = (row1["lat"], row1["lng"])
                    coordinates2 = (row2["lat"], row2["lng"])
                    dmat[i,j] = distance.distance(coordinates1, 
                                                  coordinates2).km
        dmat = np.tril(dmat.T) + dmat
        return dmat
    
    def save_city_indices(self, cities_df: pd.DataFrame):
        """save the indices of every city from cities_df to a dictionary
        
        Args:
        ---
        cities_df: DataFrame
            columns -> city: str, lat: float, lng: float

        Retruns:
        ---
        dict of str keys and int values
            index_dict[city_name] = index
        """
        index_dict = dict()
        for i, row in cities_df.iterrows():
            index_dict[row["city"]] = i
        return index_dict
    
    def get_distance(self, city1: str, city2: str):
        """look up the distance between city1 and city2 from the distance
          matrix
        
        Args:
        ---
        city1: str
            name of the first city
        city2: str
            name of the second city
            
        Retruns:
        ---
        float
            distance between city1 and city2
        """
        i = self.index_dict[city1]
        j = self.index_dict[city2]
        return self.matrix[i,j]


class Individual():
    """
    One permutation of the cities from cities_df

    Attributes:
    ---
    state: list of str
        list of city: str that describe the path
    pathlength: float
        length of path when visiting the cities given by self.state in 
         order

    Methods:
    ---
    permute_state()
        rearange order of self.state with start and end city remaining the
         same
    compute_pathlength(distance_matrix)
        compute self.pathlength with distances given in distance_matrix
    mutate()
        return a new instance of Individual which is a copy of self but 
        two random cities swapped


    """
    def __init__(self, cities_df: pd.DataFrame, start_city_name: str):
        """
        Args:
        ---
        cities_df: DataFrame
            columns -> city: str, lat: float, lng: float
        start_city_name: str
            name of city the path should start and end in
        """
        self.start_city_name = start_city_name
        cities = cities_df["city"].values.tolist()
        self.state = cities
        self.state = self.permute_state()
        self.pathlength = np.inf

    def __str__(self):
        return str(self.state)
    
    def permute_state(self):
        """
        rearange order of self.state with start and end city remaining the
         same
        
        Returns:
        ---
        list of str
            permutation of the state
        """
        start = self.start_city_name
        cities = self.state
        if start not in cities:
            raise Exception("start_city_name not in cities_df")
        cities = list(filter(lambda a: a != start, cities))
        permutation = np.random.permutation(cities).tolist()
        permutation.insert(0,start)
        permutation.append(start)
        return permutation
    
    def compute_pathlength(self, distance_matrix: DistanceMatrix):
        """
        compute the length of the path given in state

        Args:
        ---
        distance_matrix: DistanceMatrix
            Stores the distances between all cities given in the same
             cities_df as at initialization

        Returns
        ---
        float
            total distance of the path through all cities
        """
        score = 0
        for i in range(1,len(self.state)):
            d = distance_matrix.get_distance(self.state[i-1], 
                                             self.state[i])
            score += d
            
        self.pathlength = score
        return score
    
    def mutate(self):
        """return a new instance of Individual which is a copy of self but
          two random cities swapped
        
        Returns:
        ---
        Inidividual
            mutated copy of self
        """
        indices = range(1,len(self.state)-1)
        c, c_ = random.sample(indices,2)
        new = copy.deepcopy(self)
        new.state = self.state[:]
        new.state[c_] = self.state[c]
        new.state[c] = self.state[c_]
        return new
    

class Population():
    """collection of Individuals with attributes and methods regarding 
    with all
    
    Attributes:
    ---
    distance_matrix: DistanceMatrix
        matrix that holds distances between every city
    individuals: list of Individual
        collection of the individuals
    
    Methods:
    ---
    reproduce()
        mutate every individual and keep the one with the shorter 
        pathlength
    return_best()
        return individual with the lowes pathlength of the entire 
        population
    """
    def __init__(self, cities_df: pd.DataFrame, 
                 start_city_name: str, 
                 number_of_individuals: int
                 ):
        individuals = list()
        self.distance_matrix = DistanceMatrix(cities_df)
        for _ in range(number_of_individuals):
            individual = Individual(cities_df, start_city_name)
            individual.compute_pathlength(self.distance_matrix)
            individuals.append(individual)
        self.individuals = individuals
        self.avg_pathlength = 0
        self.update_avg_pathlength()

    def update_avg_pathlength(self):
        """compute the average pathlenght of all individuals in the popoulation.
        Store the value in self.avg_pathlength"""
        sum_pl = 0
        for individual in self.individuals:
            sum_pl += individual.pathlength
        self.avg_pathlength = sum_pl/len(self.individuals)

    
    def reproduce(self):
        """Use only individuals that are better than average for 
          reproduction. Make x babies from randomly chosen parents, 
          where x is the number of individuals in the old generation.
          Then mutate every baby and keep the one with the shorter 
          pathlength. 
          The babies replace the old generation.
          """
        new_generation = list()
        fertile_individuals = [ind for ind in self.individuals if 
                                ind.pathlength 
                                < self.avg_pathlength]
        if len(fertile_individuals) > 2:
            parent_pool = fertile_individuals
        else:
            parent_pool = self.individuals
        for i in range(len(self.individuals)):
            parent1, parent2 = random.sample(parent_pool, 2)
            baby = self.make_baby(parent1, parent2)
            mutation = baby.mutate()
            mutation.compute_pathlength(self.distance_matrix)
            if mutation.pathlength < baby.pathlength:
                baby = mutation
            new_generation.append(baby)
        
        self.individuals = new_generation
        self.update_avg_pathlength()

    def return_best(self):
        """return individual with the lowes pathlength of the entire
         population
        
        Retruns:
        ---
        Individual
            best individual
        """
        best = self.individuals[0]
        for individual in self.individuals[1:]:
            if individual.pathlength < best.pathlength:
                best = individual
        return best
    
    def make_baby(self, in1: Individual, in2: Individual):
        """Produce a baby individual from the two parent individual. 
          The baby is a merge of parent 1 and 2 where the fitter
          parent inherits more of its elements to the baby.
          Duplicate elements in the baby are replaced with the 
          elements that are missing, in the same order as they occur
          in parent 1.
          
        Args:
        ---
        in1: Individual
            Parent 1
        in2: Individual
            Parent 2
        
        Retruns:
        ---
        baby: Individual
            merge of Parent 1 and 2"""
        fitness1 = in2.pathlength/(in1.pathlength + in2.pathlength)
 
        path1 = in1.state[1:-1]
        path2 = in2.state[1:-1]
        split_index = int(round(fitness1*len(path1)))

        baby_path = path1[:split_index] + path2[split_index:]

        missing_cities = [c for c in reversed(path1) if c not in baby_path]
        if missing_cities != []:
            seen = []
            for i, c in enumerate(baby_path):
                if c in seen:
                    baby_path[i] = missing_cities.pop()
                else:
                    seen.append(c)
        
        baby = copy.deepcopy(in1)
        baby.state[1:-1] = baby_path
        baby.compute_pathlength(self.distance_matrix)
        return baby