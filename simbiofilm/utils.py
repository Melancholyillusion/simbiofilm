"""simbiofilm Base Classes."""

import numpy as np
from configparser import ConfigParser

from .core import Container


class ndict(dict):
    """Dictionary that allows ndict.item for getting, but not setting."""

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise KeyError(name)

    def copy(self):
        """Shallow copy of nD."""
        return ndict(super().copy())


def _convert(value):
    """Convert a string to what makes sense."""
    if not isinstance(value, str):
        return value
    try:  # float next
        return float(value)
    except ValueError:  # bool finally
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        if (value[0] + value[-1]) == "()":
            return tuple(_convert(x) for x in value[1:-1].split(","))
    return value  # remains string if nothing else works


def writecfg(filename, cfg):
    """Write cfg to filename."""
    with open(filename, "w") as f:
        for section in cfg:
            if cfg[section]:
                print("".join(["[", section, "]"]), file=f)
                for item in cfg[section]:
                    print("".join([item, " = ", str(cfg[section][item])]), file=f)
                print(file=f)


def _get_from_npz(filename):
    with np.load(filename) as f:
        configdata = f["config"]
        cfg = ndict()
        for name, value in configdata:
            section, item = name.split(":")
            if section not in cfg:
                cfg[section] = ndict()
            cfg[section][item] = _convert(value)
    return cfg


def cfg_from_dict(cfg_dict):
    cfg = ndict()
    for section in cfg_dict:
        if section.lower() in cfg:
            msg = f"Duplicate section: {section.lower()}. Already in cfg:\n{cfg}"
            raise KeyError(msg)
        cfg[section.lower()] = ndict()
        for option in cfg_dict[section]:
            if option in cfg[section.lower()]:
                msg = f"Duplicate option: {option.lower()}. Already in section: {section.lower()}"
                raise KeyError(msg)
            cfg[section.lower()][option.lower()] = _convert(cfg_dict[section][option])
    return cfg


def getcfg(filename):
    """From a filename, get a lowercase dict representing everything."""
    if filename.endswith(".npz"):
        return _get_from_npz(filename)
    cfg = ConfigParser()
    cfg.read(filename)
    return cfg_from_dict(cfg)


def parse_params(names, values, cfg=None):
    """Parse line of values, intended to be from command line.

    names is expected to be of form:
        general:seed,substrate:max,matrix:density
    The first arg per pair is the category of the config, second is the item.

    values is a string of the form:
        1,10,22e3

    If cfg is provided (from simbiofilm.getcfg), it will overwrite or add the
    values from the param line
    """
    cfg = ndict() if cfg is None else cfg.copy()
    if isinstance(names, str):
        names = names.split(',')
    names = [n.lower() for n in names]
    if isinstance(values, str):
        values = values.split(',')
    for (name, val) in zip(names, values):
        section, item = name.split(":")
        if section not in cfg:
            cfg[section] = ndict()
        cfg[section][item] = _convert(val)
    return cfg


def count(space, containers, fill_ones=False):
    """Count the number of particles at each grid point."""
    containers = to_list(containers)
    total = np.zeros(space.shape, dtype=int)
    for container in containers:
        np.add.at(total.reshape(-1), container.location, 1)
    if fill_ones:  # TODO: determine if fill_ones is necessary.
        total[total == 0] = 1
    return total


def to_grid(space, containers, param, fill_ones=False):
    """Sum the particle properties at each grid point."""
    containers = to_list(containers)
    total = np.zeros(space.shape)
    for container in containers:
        np.add.at(total.reshape(-1), container.location, getattr(container, param))
    if fill_ones:
        total[total == 0] = 1
    return total


def to_list(container):
    """Return list of single container if input is a container."""
    if isinstance(container, Container):
        return [container]
    elif container is None:
        return []
    return container


def biofilm_edge(space, biomass):
    # Returns an array where the transition from 1 to -1 is an edge.
    grid = space.meshgrid[0].copy()
    grid[biomass == 0] = 0
    height = grid.argmax(axis=0)
    if height.max() == 0:
        edge = -np.ones(space.shape)
        edge[np.where(biomass > 0)] = 1
        return edge
    height[(height == 0) & (biomass[0] == 0)] = -2
    height += 1
    nodes = (
        np.ravel_multi_index(np.where(height >= 0), space.shape[1:])
        + (height[height >= 0]) * np.prod(space.shape[1:])
    ).tolist()

    edge = np.zeros(space.shape, dtype=int)
    edge[height.max() + 1:] = -1

    edge, biomass = edge.reshape(-1), biomass.reshape(-1)
    while nodes:
        ix = nodes.pop(0)
        for ix_n in space.neighbors[ix]:
            if biomass[ix_n] == 0 and edge[ix_n] == 0:
                edge[ix_n] = -1
                nodes.append(ix_n)
    edge[edge == 0] =  1
    return edge.reshape(space.shape)
