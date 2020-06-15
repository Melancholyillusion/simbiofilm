"""Some simple events and their intended conditions."""
import numpy as np

from .utils import to_list, count


def empty_container(space, t, tcheck, containers):
    """Check for empty containers, after a given time."""
    return np.sum(count(space, containers)) == 0 and t > tcheck


def empty_after_infection(space, t, phage, biomass):
    return empty_container(space, t, 0, biomass) if phage.infected else False
