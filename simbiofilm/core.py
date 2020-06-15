"""simbiofilm Base Classes."""

import gc
import os
import csv
import numpy as np
from random import seed
import numba
import pyamg
from time import time
from collections import deque
from scipy import sparse as sp
from fipy import Grid2D, Grid3D


def _prefix_name_to_dict(name, dictionary):
    for key in list(dictionary.keys()):
        newkey = "{}_{}".format(name, key)
        dictionary[newkey] = dictionary.pop(key)
    return dictionary


def _strip_name_from_dict(name, dictionary):
    name = name + "_"
    n = len(name)
    for key in list(dictionary.keys()):
        if key.startswith(name):
            newkey = key[n:]
            dictionary[newkey] = dictionary.pop(key)
    return dictionary


def _make_time_condition(inftime):
    def infection_condition(space, time, *args, **kwargs):
        return time >= inftime

    return infection_condition


def _make_dt_constraint_function(value):
    def dt_constraint(*args, **kwargs):
        return value

    return dt_constraint


class Simulation(object):
    """Back-end for handling information, transfer of information."""

    def __init__(self, space, cfg):
        """Description.

        Parameters
        ----------
        space

        """
        self.all_end_conditions = []
        self.all_containers = []
        self.all_behaviors = []
        self.post_output = []
        self.all_events = []
        self.space = space
        self.dt = 0
        self.t = 0
        self.iteration = 0
        self.initialized = False
        self.out_dir = None
        self.config = cfg
        self._output_header = ["iteration", "time", "dt"]
        self._summary_writer = None
        self._summary_file = None
        self._save_frequency = 0
        self._event_at = []
        self._max_sim_time = cfg.general.max_sim_time if "max_sim_time" in cfg.general else None

        if 'seed' in cfg.general:
            np.random.seed(int(cfg.general.seed))
            seed(int(cfg.general.seed))
            numba.jit(lambda x: np.random.seed(x), nopython=True)(int(cfg.general.seed))

        self.forceGC = cfg.general.forcegc if "forcegc" in cfg.general else False


    def _stop_if_initialized(self):
        if self.initialized:
            msg = "Attempting to add simulation components when initialized."
            raise RuntimeError(msg)

    def add_containers(self, *args):
        """Add all containers to the list Simulation.all_containers."""
        self._stop_if_initialized()

        if not args and isinstance(args[0], list):
            args = args[0]

        for container in args:
            self.all_containers.append(container)

        uniq = set([c.name for c in self.all_containers])
        if len(uniq) < len(self.all_containers):
            msg = "Not all container names are unique: {}"
            raise ValueError(msg.format([c.name for c in self.all_containers]))

    def add_behavior(self, behavior_function, *args, dt_max=None, post_output=False):
        """Add a behavior function with args and dt constraint."""
        self._stop_if_initialized()
        if dt_max is None:
            dt_max = _make_dt_constraint_function(np.inf)
        elif not callable(dt_max):
            dt_max = _make_dt_constraint_function(dt_max)

        if post_output:
            self.post_output.append((dt_max, behavior_function, args))
        else:
            self.all_behaviors.append((dt_max, behavior_function, args))

    def add_event(self, event, condition, *args):
        """Add an event when condition is met."""
        self._stop_if_initialized()
        if not callable(condition):
            self._event_at.append(float(condition))
            condition = _make_time_condition(condition)

        self.all_events.append((event, condition, args))

    def add_end_condition(self, condition, message, *args):
        """Adds an end condition with given message."""
        self._stop_if_initialized()
        self.all_end_conditions.append((condition, message, args))

    def _check_end_conditions(self):
        endmsg = ""
        for cond, message, args in self.all_end_conditions:
            endmsg += message if cond(self.space, self.t, *args) else ""
        return endmsg

    def initialize(self, output_directory, log_header=[], save_frequency=0):
        """Initialize the containers, marking the sim ready to iterate."""
        self._setup_output(output_directory, log_header, save_frequency)
        self._event_at = sorted(self._event_at, reverse=True)
        self.initialized = True
        print("Initialized.")

    def finish(self):
        """Clean up sim I/O operations. Called by perform_iterations."""
        self._summary_file.close()

    def iterate(self, t, dt_min=0.005, dt_max=np.inf, cleanup=False):
        """Initialize the containers, marking the sim ready to iterate.

        Parameters
        ----------
        t : float
            How long to iterate

        """
        start_time = time()
        t = self.t + t
        t_msg = "End time reached:{}".format(t)
        # self.add_end_condition(lambda _, y: y >= t, t_msg)
        self.all_end_conditions.append((lambda _, y: y >= t, t_msg, []))

        while True:
            _max = dt_max if dt_max < (t - self.t) else t - self.t

            iteration_start = time()
            self._single_iteration(dt_min, _max)
            iteration_end = time()
            if (
                self.dt > 1e-5
            ):  # avoid 'zero' dt times, for events. 1e-5 is about 1 second
                self.summarize_state(iteration_end - iteration_start)

            if self._save_frequency > 0 and self.iteration % self._save_frequency == 0:
                self._save()

            # FIXME: Hack to have post output behaviors
            if self.dt > 1e-5:
                for _, behavior_fun, args in self.post_output:
                    behavior_fun(self.space, self.dt, self.log, *args)

            if self._max_sim_time and iteration_end - iteration_start >= self._max_sim_time:
                print(f"FIN-T: iteration time >= {iteration_end - iteration_start}")
                break

            message = self._check_end_conditions()
            if message:
                print(message)
                self.all_end_conditions = [
                    x for x in self.all_end_conditions if x[1] != t_msg
                ]
                break

        end_time = time()
        print("Total time: {}".format(end_time - start_time))
        if cleanup:
            self.finish()

    def _single_iteration(self, dt_min=0.005, dt_max=np.inf):
        if not self.initialized:
            msg = "Attempting to iterate before initializing simulation."
            raise RuntimeError(msg)

        self.dt = dt_max
        for dt_fun, _, args in self.all_behaviors:
            self.dt = np.min((self.dt, dt_fun(self.space, *args)))
        self.dt = max(dt_min, self.dt)  # min dt overriden by event timing
        if self._event_at:
            if self.dt > (self._event_at[-1] - self.t):
                self.dt = self._event_at[-1] - self.t
                self._event_at.pop()

        self.t = self.t + self.dt
        self.log = {}
        if self.dt > 1e-5:
            self.iteration += 1
            for _, behavior_fun, args in self.all_behaviors:
                behavior_fun(self.space, self.dt, self.log, *args)

        removal = []
        for i, (event, condition, args) in enumerate(self.all_events):
            if condition(self.space, self.t, *args):
                event(self.space, self.t, *args)
                removal.append(i)
        for i in sorted(removal, reverse=True):
            del self.all_events[i]
        if self.forceGC:
            gc.collect()

    def _setup_output(self, directory, log_header, frequency):
        """Prepare the output directory, and if supplied output frequency."""
        self.out_dir = directory
        self._save_frequency = frequency
        np.set_printoptions(precision=3, linewidth=120)

        if self.out_dir is not None:
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)

            for container in self.all_containers:
                try:
                    _, summary = container.summarize()
                except TypeError:
                    continue
                _prefix_name_to_dict(container.name, summary)
                self._output_header.extend(summary.keys())

            if log_header:
                self._output_header.extend(log_header)

            fname = "{0}/summary.csv".format(self.out_dir)
            self._summary_file = open(fname, "w")
            self._summary_writer = csv.writer(
                self._summary_file, quoting=csv.QUOTE_NONE, lineterminator="\n"
            )
            self._summary_writer.writerow(self._output_header)

    def summarize_state(self, iteration_timing=None):
        """Docstring."""
        stdout_summary = []
        stdout_summary.append("it: {}".format(self.iteration))
        if iteration_timing:
            stdout_summary.append("sim: {0:0.3f}s".format(iteration_timing))
        stdout_summary.append("t: {0:0.3f}".format(self.t))
        stdout_summary.append("dt: {0:0.3f}".format(self.dt))
        iteration_summary = {"iteration": self.iteration, "time": self.t, "dt": self.dt}

        for container in self.all_containers:
            try:
                sout, summary = container.summarize()
            except TypeError:
                continue
            stdout_summary.append(sout)
            _prefix_name_to_dict(container.name, summary)
            iteration_summary.update(summary)

        if self.log:
            iteration_summary.update(self.log)

        # get values in order for csv summary
        summary_row = [iteration_summary[name] for name in self._output_header]
        print(", ".join(stdout_summary), flush=True)
        self._summary_writer.writerow(summary_row)
        self._summary_file.flush()

    def _save(self, compressed=True):
        """Save data to npz file. Lookup numpy.savez for more information."""

        fname = "{0}/iteration{1:04}".format(self.out_dir, self.iteration)
        config = []
        for section in self.config:
            for item in self.config[section]:
                config.append(
                    (":".join((section, item)), str(self.config[section][item]))
                )

        savedata = {"config": np.array(config)}
        for container in self.all_containers:
            moredata = container.get_save_data()
            if moredata:
                _prefix_name_to_dict(container.name, moredata)
                savedata.update(moredata)

        if savedata:
            savedata["t"] = self.t
            savedata["iteration"] = self.iteration
            save = np.savez
            if compressed:
                save = np.savez_compressed
            save(fname, **savedata)
        self._summary_file.flush()

    def load(self, fname):
        """Load files matching 'prefix*'."""
        with np.load(fname, allow_pickle=True) as f:
            self.iteration = int(f["iteration"])
            self.t = float(f["t"])
            for container in self.all_containers:
                these_data = {
                    key: f[key] for key in f.keys() if key.startswith(container.name)
                }
                these_data = _strip_name_from_dict(container.name, these_data)
                container.load(these_data)


class Container:
    """Base container class."""

    def get_save_data(self, fname):
        """Get data we need to store as a dict."""
        msg = "'{}' container subclass has not implemented 'get_save_data'."
        raise NotImplementedError(msg.format(type(self).__name__))

    def load(self, data):
        """Populate container with values from data dict."""
        msg = "'{}' container subclass has not implemented 'load'."
        raise NotImplementedError(msg.format(type(self).__name__))

    def summarize(self):
        """Return dict of summarized container."""
        msg = "'{}' container subclass has not implemented 'summarize'."
        raise NotImplementedError(msg.format(type(self).__name__))


class Space(object):
    """
    Space object.

    Parameters
    ----------
    shape : tuple
        number of gridpoints in each dimension, len=ndims
    dl : float or tuple of floats
        size of the system in micrometers, y_gridsize calculated
        from the ratio given by these size.

    """

    def __init__(self, params, dl=None, well_mixed=False):
        """See Space documentation for __init__ documentation."""
        if dl is not None:
            shape = params
        else:
            shape = params.shape
            dl = params.dl
            well_mixed = params.well_mixed

        self.well_mixed = well_mixed
        self.shape = tuple(int(x) for x in shape)
        self.dl = dl
        self.dimensions = len(shape)

        if self.dimensions > 3:
            raise ValueError("dimensions must be 1, 2, or 3")

        self.sides = tuple(np.arange(side) * dl for side in shape)
        self._laplacian_cache = {}

        self.periodic = (0,) + (1,) * (self.dimensions - 1)
        self.N = np.prod(self.shape)
        if not well_mixed:
            self.dV = self.dl ** 3
            self.neighbors = self._construct_neighbors_and_laplacian()
            self.meshgrid = np.meshgrid(*self.sides, indexing="ij")
            if self.dimensions == 2:
                self.mesh = Grid2D(dx=dl, dy=dl, nx=shape[1], ny=shape[0])
            else:
                self.mesh = Grid3D(
                    dx=dl, dy=dl, dz=dl, nx=shape[1], ny=shape[0], nz=shape[2]
                )
        else:
            self.dV = np.prod(dl * np.array(self.shape))
            if len(self.shape) == 2:
                self.dV *= dl
            self.dl = np.cbrt(self.dV)

            self.shape = (1,)
            self.L = np.atleast_2d(self.shape)
            self.neighbors = np.array([0])
            self.meshgrid = [np.array([dl])]

    def construct_mesh(self, height):
        if self.dimensions == 2:
            mesh = Grid2D(dx=self.dl, dy=self.dl, nx=self.shape[1], ny=height)
        else:
            mesh = Grid3D(
                dx=self.dl,
                dy=self.dl,
                dz=self.dl,
                nx=self.shape[1],
                ny=height,
                nz=self.shape[2],
            )
        return mesh

    def _construct_neighbors_and_laplacian(self):
        Lneighbors = self._construct_laplacian(self.shape).todia()
        Lneighbors.setdiag(0, 0)
        neighbors = {}
        for k, v in Lneighbors.todok().keys():
            neighbors.setdefault(k, []).append(v)
        return neighbors

    def _construct_laplacian(self, shape):
        L = -pyamg.gallery.laplacian.poisson(shape, dtype=np.int8)
        L.data[self.dimensions][: np.prod(shape[1:])] += 1  # no flux
        L.data[self.dimensions][-np.prod(shape[1:]) :] += 1

        # for dealing with periodic boundaries
        for dim, size in enumerate(shape[1:], 1):
            ar = np.zeros(shape)
            ar.swapaxes(0, dim)[0] = 1
            ar = ar.reshape(-1)
            L.setdiag(ar, -(size - 1))
            L.setdiag(ar, size - 1)

        return L.tocsr()

    def laplacian(self, height=None):
        if height is None:
            height = self.shape[0]
        if height not in self._laplacian_cache:
            shape = (height,) + self.shape[1:]
            L = self._construct_laplacian(shape).tolil()
            L[-np.prod(shape[1:]) :] = 0
            self._laplacian_cache[height] = (L.tocsr(), shape)
        return self._laplacian_cache[height]

    def breadth_first_search(self, grid, condition=lambda x: x > 0, init=None):
        """Search the grid of space that meets a specific condition. """
        grid_flat = grid.reshape(-1)
        if init is None:
            init = range(np.prod(self.shape[1:]))
        elif str(init).lower() == "top":
            N = np.prod(self.shape)
            init = range(N - np.prod(self.shape[1:]), N)

        nodes = deque(init)
        reachable = np.ones(np.prod(self.shape), dtype=np.int8)
        while nodes:
            ix = nodes.popleft()
            if reachable[ix] != 1:
                continue  # if we've visited it already, skip it
            if not condition(grid_flat[ix]):
                reachable[ix] = 0  # mark unreachable, and a boundary
            else:
                reachable[ix] = -1  # is reachable!
                for ix_n in self.neighbors[ix]:  # add unvisited neighbors
                    if reachable[ix_n] == 1:  # is unvisited
                        nodes.append(ix_n)
        return reachable.reshape(self.shape)
