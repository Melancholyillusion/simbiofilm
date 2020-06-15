"""simbiofilm behaviors."""
from random import choice, choices
from functools import lru_cache

import numpy as np
import skfmm
import numba
import scipy as sp
import pyamg

from fipy.solvers import LinearCGSSolver, LinearGMRESSolver, LinearPCGSolver
from fipy.matrices.scipyMatrix import _ScipyMeshMatrix
from fipy import CellVariable, DiffusionTerm, ImplicitSourceTerm

from .utils import to_grid, to_list, count, biofilm_edge
from .containers import Matrix


class MonodRateReaction(object):
    def __init__(self, solute, bacteria):
        self.bacteria = bacteria
        self.solute = solute
        self._dV = solute.space.dV

    def utilization_rate(self):
        S = self.solute.value.reshape(-1)[self.bacteria.location]
        return (
            self.bacteria.mu
            * self.bacteria.yield_s
            * S
            / (S + self.solute.P.k / self.solute.P.max)
        )

    def rate_coefficient(self):
        coeff = np.zeros(self.solute.space.shape).reshape(-1)
        np.add.at(
            coeff.reshape(-1),
            self.bacteria.location,
            self.bacteria.mass * self.bacteria.mu,
        )
        return coeff


def biomass_growth(space, dt, log, container, rate):
    """Perform biomass growth on container at rate.utilization_rate()."""
    dXdt = rate.utilization_rate() * dt * container.mass
    container.mass += dXdt
    if log is not None:
        log["growth_{}".format(container.name)] = dXdt.sum()


def biomass_growth_dt(space, container, rate, std_dt=0.02, nlayers=2):
    # Mass per grid * grid points per layer * 2 layers = max mass allowed to increase
    if container.n == 0:
        return np.inf
    maxmass = container.P["density"] * space.dV * np.prod(space.shape[1:]) * nlayers
    dt_total_biofilm = maxmass / np.sum(rate.utilization_rate() * container.mass)
    return min(dt_total_biofilm, std_dt)


def bacterial_division(space, dt, log, container):
    """Divide bacteria who need it."""
    for individual in container:
        if individual.mass > individual.division_mass:
            individual.mass = individual.mass / 2
            clone = container.clone(individual)


def detach_biomass(space, dt, log, containers):
    # cutoff = ((space.meshgrid[0] / np.max(space.meshgrid[0])) < 0.1)
    containers = to_list(containers)
    biomass_list = []
    for container in containers:
        biomass_list.append(to_grid(space, containers, "mass"))
    total_biomass = np.sum(biomass_list, axis=0)
    reachable = space.breadth_first_search(total_biomass)
    isdetached = (reachable != -1).reshape(-1)

    for container in containers:
        container.multi_remove(isdetached[container.location])

    if log is not None:
        for biomass, container in zip(biomass_list, containers):
            detached = biomass.reshape(-1)[isdetached].sum()
            log["detached_{}".format(container.name)] = detached


def erode_bacteria(space, dt, log, biomass_containers, rate, topgrid=0.10):
    top = int(space.shape[0] * (1 - topgrid))
    biomass_containers = to_list(biomass_containers)
    biomass = to_grid(space, biomass_containers, "mass")

    zeros = biofilm_edge(space, biomass)

    erosion_force = np.square(space.meshgrid[0] + space.dl) * rate / space.dl
    time_to_erode = np.array(
        skfmm.travel_time(zeros, erosion_force, dx=1, periodic=space.periodic, order=1)
    )
    # TODO: use higher order. skfmm has a bug, #18

    time_to_erode[top:] = 0  # instantly erode at top
    zeros = time_to_erode == 0
    if np.any(zeros):
        time_to_erode[zeros] = dt / 1e10
    shrunken_volume = np.exp(-dt / time_to_erode).reshape(-1)
    if log is not None:
        original_mass = np.zeros(space.shape)
    for container in biomass_containers:
        if log is not None:
            original_mass[:] = to_grid(space, container, "mass")
        container.mass *= shrunken_volume[container.location]
        container.multi_remove(container.mass < (container.division_mass / 3))

        if log is not None:
            mass = original_mass - to_grid(space, container, "mass")
            log["eroded_{}".format(container.name)] = mass.sum()


def cut_max_density(space, dt, log, biomass_containers, maxvol=0.90):
    if not space.well_mixed:
        raise RuntimeError("Trying to run a well mixed behavior in a spatial system.")

    volumes = [c.volume.sum() for c in biomass_containers]
    total = np.sum(volumes)
    if total <= maxvol:
        return

    avg_volume = [np.average(c.volume) for c in biomass_containers]
    volume_tolose = volumes / total * (total - maxvol)
    ntolose = [
        ((v / avg) if avg > 0 else 0) for v, avg in zip(volume_tolose, avg_volume)
    ]
    ntolose = np.ceil(ntolose).astype(int)
    for n, container in zip(ntolose, biomass_containers):
        if n == 0:
            continue
        removals = np.zeros(container.n, dtype=bool)
        removals[np.random.choice(container.n, n, replace=False)] = True
        container.multi_remove(removals)


def detach_phage(space, dt, log, phage, biomass_containers):
    """Remove phage off biomass."""
    biomass_containers = to_list(biomass_containers)
    space = phage.space
    total_biomass = to_grid(space, biomass_containers, "mass")

    reachable = space.breadth_first_search(total_biomass)
    isdetached = (reachable != -1).reshape(-1)

    phage.multi_remove(isdetached[phage.location])


def lysis(space, dt, log, phage, infected):
    """Test."""
    if infected.n == 0:  # TODO: check for n == 0 in setattr
        return
    infected.incubation_time -= dt

    for individual in infected:
        if individual.incubation_time <= 0:
            params = {}
            for name in individual.dtype.names:
                if "phage_" in name:
                    params[name[6:]] = getattr(individual, name)
                params["remainder"] = -individual.incubation_time / dt

            phage.add_multiple_individuals(
                int(individual.phage_burst), individual.location, params
            )
            infected.remove(individual)
    phage.remainder = np.clip(phage.remainder, 0, None)


def lysis_dt(space, phage, infected, substrate=None):
    """Minimum of the incubation times left on the infected."""
    # return np.inf if infected.n == 0 else max(np.min(infected.incubation_time), 0.005)
    return np.inf if infected.n == 0 else np.min(infected.incubation_time)


def _build_biomass(space, biomass_containers):
    """Generates {location: [(container, particle, volume_p), (c, p, v) ...], location: [...]}."""
    biomass = {}
    for container in biomass_containers:
        rate = container.slow * container.mass
        for particle, rate in zip(container, rate):
            if particle.location in biomass:
                biomass[particle.location].append((container, particle, rate))
            else:
                biomass[particle.location] = [(container, particle, rate)]
    return biomass


def phage_interaction_wellmixed(space, dt, log, phage, infected, biomass_containers):
    """Infect susceptible bacteria, without resistance."""
    biomass_containers = to_list(biomass_containers)
    biomass = _build_biomass(space, biomass_containers)

    slows = np.zeros(space.shape)
    for cntnr in biomass_containers:
        np.add.at(
            slows.reshape(-1), cntnr.location, cntnr.slow * cntnr.mass / space.dV * dt
        )

    interaction_rate = np.sum([c.slow * c.mass / space.dV for c in biomass_containers])
    time_to_interact = np.log(1 - np.random.rand(phage.n)) / interaction_rate
    adsorptions = np.random.rand(phage.n) < 1 - np.exp(
        -phage.adsorption_rate * (dt - time_to_interact)
    )  # (dt - time_to_interact) is positive where interactions fail. exp(positive) > 1

    biomass = _build_biomass(space, biomass_containers)[0]
    p = np.array([r for _, _, r in biomass])
    p /= p.sum()
    interactions = choices(biomass, [v for c, p, v in biomass], k=adsorptions.sum())

    removals = np.zeros(phage.n, dtype=bool)
    bremovals = {c: [] for c in biomass_containers}
    for i, phg, (c, bac, _) in zip(
        np.arange(phage.n)[adsorptions], phage[adsorptions], interactions
    ):
        bac_infected, phg_removed = c.phage_interacted(bac, phg)
        # Multiple phage can infect the same thing, so have to make sure its not removed yet
        if bac_infected and bac.id not in bremovals[c]:
            bremovals[c].append(bac.id)
            infected.infect(bac, phg)
        removals[i] = phg_removed

    for container, ids in bremovals.items():
        for id in ids:
            container.remove(container.with_id(id))

    phage.multi_remove(removals)


def phage_interaction(space, dt, log, phage, infected, biomass_containers):
    """Infect susceptible bacteria, without resistance."""
    biomass_containers = to_list(biomass_containers)
    biomass = _build_biomass(space, biomass_containers)

    interactions = [ choices(population=biomass[loc], weights=[v for c, p, v in biomass[loc]])[0]
        if loc in biomass
        else (False, False, False)
        for loc in phage.location
    ]

    adsorptions = np.random.rand(phage.n) < (
        1 - np.exp(-phage.adsorption_rate * phage.remainder * dt)
    )

    removals = np.zeros(phage.n, dtype=bool)
    bremovals = {c: [] for c in biomass_containers}
    for (i, phg), (c, bac, _), ads in zip(enumerate(phage), interactions, adsorptions):
        if not c or not ads:  # objects are true, False is not.
            continue
        bac_infected, phg_removed = c.phage_interacted(bac, phg)
        # Multiple phage can infect the same thing, so have to make sure its not removed yet
        if bac_infected and bac.id not in bremovals[c]:
            bremovals[c].append(bac.id)
            infected.infect(bac, phg)
        removals[i] = phg_removed

    for container, ids in bremovals.items():
        for id in ids:
            container.remove(container.with_id(id))

    phage.multi_remove(removals)


def infect_dt(space, phage, infected, bacteria_containers):
    """Minimum of half the phage incubation period, if any phage exist."""
    return np.inf if phage.n == 0 else np.min(phage.incubation_period / 2)


def phage_randomwalk(space, dt, log, phage, biomass_containers, rate):
    """Move them, go."""

    if phage.n == 0:
        return

    biomass_containers = to_list(biomass_containers)
    step_dt = (2 * space.dl * space.dl) / phage.diffusivity
    nsteps = dt / step_dt * phage.remainder

    biomass = to_grid(space, biomass_containers, "mass")
    distance = -np.array(
        skfmm.distance(biofilm_edge(space, biomass), dx=space.dl), dtype=float
    )
    distance[distance < 0] = 0
    # rem = 1 - np.exp(-rate * step_dt * distance ** 2)
    remove = 1 - np.exp(-rate * 2 * space.dl ** 2 * distance ** 2 / phage.P.diffusivity)

    slows = np.zeros(space.shape)
    for cntnr in biomass_containers:
        np.add.at(
            slows.reshape(-1), cntnr.location, cntnr.slow * cntnr.mass / space.dV
        )
    location = np.array(np.unravel_index(phage.location, space.shape)).T

    if space.dimensions == 2:
        location, remainder = _rw_stuck_2D(
            location, nsteps, step_dt, slows, np.array(space.shape), remove
        )
    elif space.dimensions == 3:
        location, remainder = _rw_stuck_3D(
            location, nsteps, step_dt, slows, np.array(space.shape), remove
        )

    remainder[nsteps > 0] = remainder[nsteps > 0] / nsteps[nsteps > 0]
    phage.location = np.ravel_multi_index(location.T, space.shape)
    phage.remainder = remainder


@numba.jit(nopython=True, parallel=True)
def _rw_stuck_2D(location, nsteps, step_dt, slows, shape, premoval):
    remainder = np.zeros_like(nsteps)
    for i in numba.prange(location.shape[0]):
        loc = location[i]
        for step in range(nsteps[i]):

            # sp.special.erf(1/(np.sqrt(2 * np.pi))) for 2D
            if np.random.rand() < 0.42737488393046696:
                continue

            if np.random.rand() < premoval[loc[0], loc[1]]:
                break

            loc_ = loc.copy()
            loc_[np.random.randint(2)] += -1 if np.random.rand() < 0.5 else 1

            # check for boundaries. Bump into wall and fail to move.
            if loc_[0] == shape[0] or loc_[0] == -1:
                continue

            # check for periodicity
            for dim in range(1, len(loc_)):
                if loc_[dim] == shape[dim] or loc_[dim] == -1:
                    loc_[dim] = loc_[dim] % shape[dim]

            sl = slows[loc[0], loc[1]] + slows[loc_[0], loc_[1]]
            P = 1 - np.exp(-sl * step_dt[i])
            if np.random.rand() < P:  # interact
                break

            # make the move!
            loc = loc_
        remainder[i] = nsteps[i] - step

        location[i] = loc
    return (location, remainder)


@numba.jit(nopython=True, parallel=True)
def _rw_stuck_3D(location, nsteps, step_dt, slows, shape, premoval):
    remainder = np.zeros_like(nsteps)
    for i in numba.prange(location.shape[0]):
        loc = location[i]
        for step in range(nsteps[i]):

            # sp.special.erf(1/(2 * np.sqrt(2 * np.pi))) for 3D
            if np.random.rand() < 0.22212917330884568:
                continue

            if np.random.rand() < premoval[loc[0], loc[1], loc[2]]:
                break

            loc_ = loc.copy()
            loc_[np.random.randint(3)] += -1 if np.random.rand() < 0.5 else 1

            # check for boundaries. Bump into wall and fail to move.
            if loc_[0] == shape[0] or loc_[0] == -1:
                continue

            # check for periodicity
            for dim in range(1, len(loc_)):
                if loc_[dim] == shape[dim] or loc_[dim] == -1:
                    loc_[dim] = loc_[dim] % shape[dim]

            sl = slows[loc[0], loc[1], loc[2]] + slows[loc_[0], loc_[1], loc_[2]]
            P = 1 - np.exp(-sl * step_dt[i])
            if np.random.rand() < P:  # interact
                break

            # make the move!
            loc = loc_
        remainder[i] = nsteps[i] - step

        location[i] = loc
    return (location, remainder)


@numba.jit(nopython=True)
def _distance(p1, p2, shape):
    total = 0
    for i, (a, b) in enumerate(zip(p1, p2)):
        delta = abs(b - a)
        if i == 0:  # no periodic y
            total += delta ** 2
            continue
        if delta > shape[i] - delta:
            delta = shape[i] - delta
        total += delta ** 2
    return np.sqrt(total)


def relax_line_shove(space, dt, log, biomass_containers):
    """Shove overfull grid points along the path to the nearest empty point."""
    maxvol = 1  # packing ratio?

    biomass_containers = to_list(biomass_containers)
    location, volume = _unpack_containers(
        space, biomass_containers, ["location", "volume"]
    )
    total_volume = np.zeros(space.shape)
    np.add.at(total_volume.reshape(-1), location, volume)

    while True:
        updates = list(_get_updates(total_volume.reshape(-1), maxvol))
        shoved = set()
        if not updates:
            break

        while updates:
            ix = updates.pop()
            path = _get_shove_path(total_volume, ix)

            if any(ix in shoved for ix in path):
                break

            shoved.update(path)
            layer_size = np.prod(space.shape[1:])
            _shove_along_path(
                path, location, volume, total_volume.reshape(-1), layer_size
            )

    _repack_containers(location, biomass_containers)


def _unpack_containers(space, biomass_containers, params):
    return [
        np.concatenate([c[param] for c in biomass_containers if c.n > 0])
        for param in params
    ]


def _get_shove_path(volumes, ix):
    loc = list(np.unravel_index(ix, volumes.shape))
    if len(loc) == 2:
        candidates = _get_candidates_2d(volumes, loc)
    if len(loc) == 3:
        candidates = _get_candidates_3d(volumes, loc)
    target = list(candidates[np.random.randint(len(candidates))])
    shift = []
    for i in range(1, len(volumes.shape)):
        shift.append(volumes.shape[i] // 2 - loc[i])
        loc[i] = (loc[i] + shift[i-1]) % volumes.shape[i]
        target[i] = (target[i] + shift[i-1]) % volumes.shape[i]
    path = _get_line(loc, target)
    for i in range(1, len(volumes.shape)):
        path[:, i] -= shift[i-1]
        path[:, i] %= volumes.shape[i]
    return np.ravel_multi_index(list(zip(*path)), volumes.shape)


@numba.njit
def _get_candidates_2d(volumes, ix):
    yinit, xinit = ix
    explored = np.zeros_like(volumes, dtype=np.bool8)
    node = [(yinit, xinit)]

    candidates = []
    candistance = volumes.shape[0] ** 2 + volumes.shape[1] ** 2
    while node:
        _ix = node.pop(0)
        y, x = _ix
        if explored[y, x]:
            continue
        explored[y, x] = True
        if _distance(ix, _ix, volumes.shape) > candistance:
            continue

        if volumes[y, x] <= 0.01:
            candidates.append(_ix)
            candistance = _distance(ix, _ix, volumes.shape)
            continue

        for j, i in ((1, 0), (0, -1), (0, 1), (-1, 0)):
            _y, _x = y + j, x + i
            if _y >= 0 and _y < volumes.shape[0]:
                if _x == -1:
                    _x = volumes.shape[1] - 1
                elif _x == volumes.shape[1]:
                    _x = 0
                node.append((_y, _x))
    return candidates


@numba.njit
def _get_candidates_3d(volumes, ix):
    yinit, xinit, zinit = ix
    explored = np.zeros_like(volumes, dtype=np.bool8)
    node = [(yinit, xinit, zinit)]

    candidates = []
    candistance = volumes.shape[0] ** 2 + volumes.shape[1] ** 2 + volumes.shape[2] ** 2
    while node:
        _ix = node.pop(0)
        y, x, z = _ix
        if explored[y, x, z]:
            continue
        explored[y, x, z] = True
        if _distance(ix, _ix, volumes.shape) > candistance:
            continue

        if volumes[y, x, z] <= 0.01:
            candidates.append(_ix)
            candistance = _distance(ix, _ix, volumes.shape)
            continue

        for j, i, k in (
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, -1),
        ):
            _y, _x, _z = y + j, x + i, z + k
            if _y >= 0 and _y < volumes.shape[0]:
                if _x == -1:
                    _x = volumes.shape[1] - 1
                elif _x == volumes.shape[1]:
                    _x = 0
                if _z == -1:
                    _z = volumes.shape[1] - 1
                elif _z == volumes.shape[1]:
                    _z = 0
                node.append((_y, _x, _z))
    return candidates


def _get_line(start, end):
    start = np.array(start, dtype=int)
    end = np.array(end, dtype=int)
    n_steps = np.max(np.abs(end - start)) + 1
    dim = start.size
    slope = (end - start).astype(float)
    scale = np.max(np.abs(slope))
    if scale != 0:
        slope = slope / scale
    stepmat = np.arange(n_steps).reshape((n_steps, 1)).repeat(dim, axis=1)
    return np.rint(start + slope * stepmat).astype(int)


@numba.njit
def _shove_along_path(path, location, volume, total_volume, layersize):
    for start, target in zip(path[:-1], path[1:]):
        y, yt = start // layersize, target // layersize
        at_start_indices = np.where(location == start)[0]
        np.random.shuffle(at_start_indices)  # helps to prevent bias
        for ix in at_start_indices:
            if volume[ix] == 0:
                location[ix] = target
            if total_volume[start] > 1:
                total_volume[start] -= volume[ix]
                location[ix] = target
                total_volume[target] += volume[ix]


def _repack_containers(location, biomass_containers):
    count = 0
    for container in biomass_containers:
        container.location = location[count : count + container.n]
        count += container.n


@numba.jit(nopython=True)
def _distance(p1, p2, shape):
    total = 0
    for i, (a, b) in enumerate(zip(p1, p2)):
        delta = abs(b - a)
        if i == 0:  # no periodic y
            total += delta ** 2
            continue
        if delta > shape[i] - delta:
            delta = shape[i] - delta
        total += delta ** 2
    return np.sqrt(total)


def _get_updates(flat_volume, maxvol):
    updates = np.where(flat_volume > maxvol)[0]
    updates = [(ix, flat_volume[ix]) for ix in updates]
    return [x[0] for x in sorted(updates, key=lambda x: 1 / x[1])]


class QSSDiffusionReaction:
    """Quasi-steady-state diffusion reaction solver. """

    def __init__(
        self, space, solute, ratereactions, max_coarse=10, maxiter=100, tol=1e-8
    ):
        # where do we put the mesh and constraining?
        # mesh goes in Space. constraints go in solute.
        self.solute = solute
        self.diffusivity = solute.P.diffusivity * solute.P.max
        self.space = space
        self.reactions = ratereactions
        self.solver = LinearGMRESSolver()  # LinearGMRESSolver is default
        self._layer = np.prod(self.space.shape[1:])
        self._layer_height = round(solute.P.h / solute.space.dl) + 1
        self._sr = solute.k / solute.max
        self._tolerance = tol

    def __call__(self, space, dt, log, active_biomass):
        height = np.max(np.where(to_grid(self.space, active_biomass, "mass") > 0)[0])
        equation, phi, size = self._setup_equation(height)
        while equation.sweep(var=phi, solver=self.solver) > self._tolerance:
            phi.updateOld()
        self.solute.value.reshape(-1)[size:] = 1
        self.solute.value.reshape(-1)[:size] = phi.value

    def _setup_equation(self, biomass_height):
        height = biomass_height + self._layer_height
        size = height * self._layer
        mesh = self.space.construct_mesh(height)
        variables, terms = [], []
        phi = CellVariable(name=self.solute.name, mesh=mesh, hasOld=True)
        for r in self.reactions:
            variables.append(
                CellVariable(name=f"{r.bacteria.name}_rate", mesh=mesh, value=0.0)
            )
            terms.append(ImplicitSourceTerm(coeff=(variables[-1] / (phi + self._sr))))
        equation = DiffusionTerm(coeff=self.diffusivity) - sum(terms)
        phi.constrain(1, where=mesh.facesTop)

        for var, coef in zip(
            variables, [r.rate_coefficient()[:size] for r in self.reactions]
        ):
            try:
                var.setValue(coef / self.space.dV)
            except ValueError as err:
                print("Boundary layer height greater than system size")
                raise err
        phi.setValue(self.solute.value.reshape(-1)[:size])
        return equation, phi, size

