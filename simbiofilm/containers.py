"""simbiofilm Container Classes."""
from itertools import count
import numpy as np
from .core import Container


class ParticleContainer(Container):
    """Contains full particle information. If location is int, old 'hybrid'.

    Parameters
    ----------
    name : str
        Unique name of the container, identical names will throw an error
        when added to the Simulation.
    space : Space
        Space object for the simulation.
    params : ParameterSet
        ParameterSet populated with parameters fulfilling the associated
        behaviors' requirements
    maxn : int, optional
        Initial maximum size of the data array. The array will double in
        size when needed.

    """

    _unique_id = count(1)

    def __init__(self, name, space, params, maxn=1024):
        """See ParticleContainer for init documentation."""
        datatypes = [("id", "u4")]
        datatypes.append(("parent", "u4"))
        datatypes.append(("mass", float))
        datatypes.append(("location", "u4"))
        for k, v in params.items():
            if k not in list(zip(*datatypes))[0]:
                datatypes.append((k, type(v)))

        super().__setattr__("_params", dict(datatypes))
        self._data = np.rec.array(np.zeros(int(maxn), dtype=datatypes))
        self._view_cache = {}
        self._id_to_index = {}
        self._count = 0
        self._current_id = 1

        self.name = name
        self.space = space
        self.P = params

    def __getitem__(self, name):
        return getattr(self, name)

    def __getattr__(self, name):
        if name in self._params:
            if name not in self._view_cache:
                self._view_cache[name] = getattr(self._data, name)[: self.n]
            return self._view_cache[name]
        msg = "'{}' has no attribute or parameter '{}'"
        msg = msg.format(type(self).__name__, name)
        raise AttributeError(msg)

    def __setattr__(self, name, value):
        if name in self._params:
            self._data[name][: self.n] = value
        else:
            super().__setattr__(name, value)

    def __iter__(self):
        """Provide a view of the record."""
        for ind in reversed(self._data[: self.n]):
            yield ind

    def _clear_cache(self):
        self._view_cache = {}

    def _add_particle(self, params):
        if self._count == self._data.shape[0]:
            self._data = np.concatenate((self._data, self._data))
            self._data = np.rec.array(self._data)

        id = next(self._unique_id)
        self._current_id = id  # for saving
        ix = self._count
        self._id_to_index[id] = ix
        self._count += 1
        for k, v in params.items():
            setattr(self._data[ix], k, v)
        self._data[ix].id = id
        self._clear_cache()
        return self._data[ix]

    def with_id(self, id):
        """Return the record of the particle with given id."""
        if id in self._id_to_index:
            return self._data[self._id_to_index[id]]
        msg = "'{}' has no particle with id '{}'"
        msg = msg.format(type(self).__name__, id)
        raise KeyError(msg)

    def at_location(self, location):
        """Return the record of the particle nearest to location."""
        if self.n == 0:
            return
        ixs = np.where(self.location == location)[0]
        for ix in ixs:
            yield self._data[ix]

    def add_individual(self, location, params=None):
        """Add an individual with default parameters at location."""
        if hasattr(location, "__iter__"):
            location = np.ravel_multi_index(location, self.space.shape)

        all_params = {}
        all_params.update(self.P.items())
        if params:
            all_params.update(params.items())
        all_params["location"] = location
        return self._add_particle(all_params)

    def add_multiple_individuals(self, n, location, params=None):
        """Add multiple individuals. See add_individual."""
        return [self.add_individual(location, params) for _ in range(n)]

    def clone(self, parent):
        """Clones parent exactly."""
        parameters = dict(zip(parent.dtype.names, parent))
        return self.add_individual(parent.location, parameters)

    def remove(self, individual):
        """Kill individual."""
        if not individual == self.with_id(individual.id):
            # TODO: better message
            msg = "Individual does not appear to be in this container."
            raise RuntimeError(msg)
        id = individual.id
        self._count -= 1
        old_ix = self._id_to_index[id]
        if old_ix < self._count:
            self._id_to_index[self._data[self._count].id] = old_ix
            self._data[old_ix] = self._data[self._count]
        del self._id_to_index[id]
        self._clear_cache()

    def multi_remove(self, removal):
        """Kill individual."""
        if len(removal) != self.n:
            msg = "length of array for multi_remove must equal n of container."
            raise RuntimeError(msg)
        if np.sum(removal) == 0:
            return

        nalive = self._count - removal.sum()
        if nalive > 0:
            alive_ix = np.array(
                [x for x in np.where(np.logical_not(removal))[0] if x >= nalive]
            )
            dead_ix = np.array([x for x in np.where(removal)[0] if x < nalive])
            if alive_ix.size != dead_ix.size:
                raise IndexError("This should not happen.")  # I don't think it does.
            if alive_ix.size == 0:
                self._count = nalive
                self._clear_cache()
                return  # removed only the trailing end in the data array.

            for aix, dix in zip(alive_ix, dead_ix):
                self._id_to_index[self._data[aix].id] = dix
                del self._id_to_index[self._data[dix].id]

            try:
                self._data[dead_ix] = self._data[alive_ix]
            except IndexError as err:
                print(dead_ix, alive_ix, sep="\n")
                raise err

        self._count = nalive
        self._clear_cache()

    def get_save_data(self):
        """Get savedata in a dict."""
        savedata = {}
        savedata["data"] = self._data[: self.n]
        savedata["n"] = self.n
        savedata["params"] = tuple(self._params.items())
        savedata["nextid"] = self._current_id
        return savedata

    def load(self, data):
        """Load data from dict."""
        datatypes = [(n, t) for n, t in data["params"]]
        self._params = dict(datatypes)
        self._count = int(data["n"])
        self._unique_id = count(int(data["nextid"]))

        if data["data"].shape[0] > 0:
            size = int(2 ** np.ceil(np.log2(data["data"].shape[0])))
            size = max((size, 1024))
            self._data = np.rec.array(np.zeros(size, dtype=datatypes))
            self._data[: self._count] = data["data"]
        else:
            self._data = np.rec.array(np.zeros(1024, dtype=datatypes))

        for ix in range(self.n):
            self._id_to_index[self._data[ix].id] = ix
        self._clear_cache()

    def summarize(self):
        """Summarize container."""
        mass = np.sum(self.mass)
        msg = "{0} N: {1}, mass: {2:0.3e}"
        msg = msg.format(self.name, self.n, mass)
        return msg, {"N": self.n, "mass": mass}

    @property
    def n(self):
        """Return the number of particles."""
        return self._count

    @property
    def volume(self):
        """Relative grid volume, NOT absolute volume."""
        return (
            np.zeros(self.n)
            if "density" not in self.P
            else self.mass / (self.density * self.space.dV)
        )

    def phage_interacted(self, particle, phage):
        # Return [bool, bool] for whether or not [particle infected, phg removed]
        # [True, False] should not happen ever.
        msg = f"phage_interacted not implemented for {self}"
        raise NotImplementedError(msg)


class Bacteria(ParticleContainer):
    """Class."""

    def __init__(self, name, space, params, maxn=1024):
        """See Bacteria docstring for init documentation."""
        super().__init__(name, space, params, maxn)

    def phage_interacted(self, bacterium, phage):
        # see simbiofilm.behaviors.phage_interaction
        # Return true if we should remove the particle
        if not bacterium == self.with_id(bacterium.id):
            msg = "Individual does not appear to be in this container."
            raise RuntimeError(msg)
        return [not bacterium.resistant, bacterium.adsorbable]


class Matrix(ParticleContainer):
    """Class."""

    def __init__(self, name, space, params, sticky=False, maxn=1024):
        """See Bacteria docstring for init documentation."""
        super().__init__(name, space, params, maxn)
        self.sticky = sticky

    def phage_interacted(self, bacterium, phage, infected):
        return self.sticky


class InfectedBacteria(Bacteria):
    """Contains information about previous clean bacteria and the infecting phage."""

    def __init__(self, name, space, params, phage_params, maxn=1e4):
        """See InfectedBacteria for init documentation."""
        infected_params = params.copy()
        for key in phage_params:
            newkey = "phage_{}".format(key)
            if newkey in infected_params:
                msg = "{} in infected and phage parameters, rename one."
                raise KeyError(msg.format(newkey))
            infected_params[newkey] = phage_params[key]
        infected_params["infectedby"] = -1
        infected_params["incubation_time"] = infected_params["phage_incubation_period"]
        super().__init__(name, space, infected_params, maxn)
        self.phage_params = tuple(phage_params.keys())
        self.bacteria_params = tuple(k for k in params.keys() if k != "multi_infect")

    def infect(self, bacterium, phage):
        """."""
        pname = "phage_{}"
        params = self.P.copy()
        for p in self.bacteria_params:
            params[p] = getattr(bacterium, p)
        for phg_p, p in ((pname.format(x), x) for x in self.phage_params):
            params[phg_p] = getattr(phage, p)
        params["infectedby"] = phage.id
        params["incubation_time"] = (
            phage.incubation_period * phage.remainder * np.random.normal(1, 0.1)
        )

        return self.add_individual(bacterium.location, params)

    def summarize(self):
        """Summarize infected bacteria."""
        msg = "{0} N: {1}"
        msg = msg.format(self.name, self.n)
        return msg, {"N": self.n}

    def phage_interacted(self, bacterium, phage):
        # see simbiofilm.behaviors.phage_interaction
        # Return true if we should remove the particle
        return [False, bacterium.multi_infect]


class Phage(ParticleContainer):
    """Class."""

    def __init__(self, name, space, params, maxn=1e5):
        """See Phage for init documentation."""
        params["remainder"] = float(1)
        super().__init__(name, space, params, maxn)
        self.infected = False

    def add_individual(self, index, params=None):
        """Add an individual with default parameters at location."""
        if params is None:
            params = self.P
        super().add_individual(index, params)

    def get_save_data(self):
        savedata = super().get_save_data()
        savedata["infected"] = self.infected
        return savedata

    def load(self, data):
        self.infected = data["infected"]
        del data["infected"]
        super().load(data)

    def summarize(self):
        """Summarize phage."""
        self.remainder = 1
        msg = "{0} N: {1}"
        msg = msg.format(self.name, self.n)
        return msg, {"N": self.n}


class Solute(Container):
    """Container for soluble diffusible material."""

    def __init__(self, name, space, params, dtype=float):
        self.name = name
        self.space = space
        self.value = np.zeros(space.shape, dtype)
        self.P = params

    def __getattr__(self, name):
        if name in self.P:
            return getattr(self.P, name)
        else:
            msg = "'{}' object has no attribute '{}'".format(type(self), name)
            raise AttributeError(msg)

    def get_save_data(self):
        return {"value": self.value}

    def load(self, data):
        self.value = data["value"]

    def summarize(self):
        return None
