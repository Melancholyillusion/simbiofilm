"""Some simple events and their intended conditions."""
import numpy as np

from .utils import to_grid


def inoculate_at(space, time, bacteria_container, n):
    """Intended to be called on empty bacteria containers."""
    params = bacteria_container.P.copy()
    mass = params.mass
    for ix in np.random.randint(np.prod(space.shape[1:]), size=n):
        params["mass"] = np.random.normal(mass, mass * 0.2)
        bacteria_container.add_individual(ix, params)


def infect_at(space, time, phage_container, biomass_containers, N):
    """Initialize phage infection."""
    biomass = to_grid(space, biomass_containers, "mass")

    reachable = space.breadth_first_search(biomass, lambda x: x == 0, "top")
    indices = list(np.where(reachable == 0))
    indices[0] += 3
    indices = np.ravel_multi_index(indices, space.shape)
    for ix in np.random.choice(indices, N):
        phage_container.add_individual(ix)


def infect_point(space, time, count, phage_container, biomass_containers):
    """Initialize phage infection."""
    if count == 0:
        return
    biomass = to_grid(space, biomass_containers, "mass")

    phage_container.infected = True

    if space.well_mixed:
        phage_container.add_multiple_individuals(count, 0)
        return

    reachable = space.breadth_first_search(biomass, lambda x: x == 0, "top")
    indices = list(np.where(reachable == 0))
    maxa = np.argmax(indices[0])

    ix = [ind[maxa] for ind in indices]
    phage_container.add_multiple_individuals(
        int(count), np.ravel_multi_index(ix, space.shape)
    )


def initialize_bulk_substrate(space, time, solute, minval=0.75):
    """Initialize bulk substrate, approximate downward."""
    mgy = space.meshgrid[0]
    solute.value = np.clip(minval * (1 + mgy / np.max(mgy)), 0, 1)
