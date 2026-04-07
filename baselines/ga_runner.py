import time
import pandas as pd
from pyCombinatorial.algorithm import genetic_algorithm
from pyCombinatorial.utils import util


def load_coordinates_from_url(url):
    df = pd.read_csv(url, sep="\t")
    return df.values


def build_distance_matrix(coordinates):
    return util.build_distance_matrix(coordinates)


def run_ga(distance_matrix, parameters):
    t0 = time.perf_counter()
    route, distance = genetic_algorithm(distance_matrix, **parameters)
    elapsed = time.perf_counter() - t0
    return route, distance, elapsed