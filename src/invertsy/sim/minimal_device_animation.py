from collections import namedtuple

from invertsy.__helpers import __root__
from invertsy.agent.agent import RouteFollowingAgent, VisualProcessingAgent, CentralComplexAgent

from ._helpers import *
from .simulation import RouteSimulation, NavigationSimulation, SimulationBase
from .simulation import PathIntegrationSimulation, TwoSourcePathIntegrationSimulation
from .minimal_device_simulation import MinimalDevicePathIntegrationSimulation
from .simulation import VisualNavigationSimulation, CentralPointNavigationSimulationBase
from .animation import AnimationBase

from scipy.spatial.transform import Rotation as R
from matplotlib import animation
from matplotlib.path import Path

import loguru as lg
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import sys
import os


class MinimalDevicePathIntegrationAnimation(AnimationBase):

    def __init__(self, sim, show_history=True, cmap="coolwarm", *args, **kwargs):
        """
        Animation for the path integration simulation. Shows the POL neurons responses in the Dorsal Rim Area, the
        position and history of positions of the agent on the map (with vegetation if provided) and the responses of
        the CX neurons (and their history if requested).

        Parameters
        ----------
        sim: PathIntegrationSimulation, TwoSourcePathIntegrationSimulation
            the path integration simulation isnstance
        show_history: bool, optional
            if True, it shows the history instead of just the current responses. Default is True
        cmap: str, optional
            the colour map for the responses of the POL neurons. Default is 'coolwarm'
        """
        kwargs.setdefault('fps', 100)
        super().__init__(sim, *args, **kwargs)

        if show_history:
            mosaic = """
                ACCCBBBB
                DDDDBBBB
                EEEEBBBB
                FFFFBBBB
                GGGGBBBB
                """
            if isinstance(sim, TwoSourcePathIntegrationSimulation) or isinstance(sim.agent, RouteFollowingAgent):
                mosaic += """HHHHBBBB
                """
            ax_dict = self.fig.subplot_mosaic(mosaic)
        else:
            ax_dict = self.fig.subplot_mosaic(
                """
                AB
                AB
                AB
                """
            )

        if isinstance(sim, MinimalDevicePathIntegrationSimulation):
            nest = sim.central_point[:2]
            feeders = [sim.distant_point[:2]]
            route = sim.route
        else:
            nest = None
            feeders = None
            route = None

        line_c, line_b, pos, self._marker = create_map_axis(world=sim.world, ax=ax_dict["B"],
                                                            nest=nest, feeders=feeders)[:4]
        vec, mbon = None, None
        if show_history:
            omm = create_dra_axis(sim.compass_sensor, cmap=cmap, ax=ax_dict["A"])
            direction = create_direction_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap, ax=ax_dict["C"])
            memory = create_memory_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap, ax=ax_dict["D"])
            sigmoid_neuron = create_sigmoid_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap, ax=ax_dict["E"])
            steering = create_steering_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap, ax=ax_dict["F"])
            steering_diff = create_steering_diff_history(self.nb_frames, sep=route.shape[0], ax=ax_dict["G"])

            if isinstance(sim, TwoSourcePathIntegrationSimulation):
                vec = create_vec_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap, ax=ax_dict["H"])
            if isinstance(sim.agent, RouteFollowingAgent):
                mbon = create_mbon_history(sim.agent, self.nb_frames, sep=route.shape[0], cmap=cmap, ax=ax_dict["H"])
        else:
            omm, direction, cl1, steering, memory, cpu4mem = create_bcx_axis(sim.agent, cmap=cmap, ax=ax_dict["A"])
            sigmoid_neuron = steering
            steering_diff = steering[0] - steering[1]

        plt.tight_layout()

        self._lines.extend([omm, direction, sigmoid_neuron, memory, steering, steering_diff, line_c, line_b, pos, vec, mbon])

        omm.set_array(sim.r_direction)
        self._show_history = show_history

    def _animate(self, i):
        """
        Runs the current iteration of the simulation and updates the data from the figure.

        Parameters
        ----------
        i: int
            the current iteration number
        """

        if isinstance(self.sim, MinimalDevicePathIntegrationSimulation):
            route_size = self.sim.route.shape[0]
        else:
            route_size = -1
            lg.logger.debug("None INSTANCE!")
        if i == 0:
            self.line_b.set_data([], [])
            # self.sim.reset(nb_samples_calibrate=10)
            self.sim.reset()
        elif "xyz_out" in self.sim.stats:
            self.line_b.set_data(np.array(self.sim.stats["xyz_out"])[..., 1],
                                 np.array(self.sim.stats["xyz_out"])[..., 0])
            # self.sim.init_inbound()

        time = self.sim.step(i)

        self.omm.set_array(np.array(self.sim.stats["direction"][-1]))

        if self._show_history:
            direction = np.zeros((self.sim.r_direction.shape[0], self.nb_frames), dtype=float)
            direction[:, :i+1] = np.array(self.sim.stats["direction"]).T
            self.direction.set_array(direction)
            memory = np.zeros((self.sim.r_memory.shape[0], self.nb_frames), dtype=float)
            memory[:, :i+1] = np.array(self.sim.stats["memory"]).T
            self.memory.set_array(memory)
            sigmoid_neuron = np.zeros((self.sim.r_sigmoid_neuron.shape[0], self.nb_frames), dtype=float)
            sigmoid_neuron[:, :i+1] = np.array(self.sim.stats["sigmoid_neuron"]).T
            self.sigmoid_neuron.set_array(sigmoid_neuron)
            steering = np.zeros((self.sim.r_steering.shape[0], self.nb_frames), dtype=float)
            steering[:, :i+1] = np.array(self.sim.stats["steering"]).T
            self.steering.set_array(steering)
            steering_diff = np.zeros(self.nb_frames, dtype=float)
            steering_diff[:i+1] = self.sim.stats["steering_diff"]
            self.steering_diff.set_data(range(self.nb_frames),steering_diff)

            if self.vec is not None:
                vec = np.zeros((self.sim.r_vec.shape[0], self.nb_frames), dtype=float)
                vec[:, :i+1] = np.array(self.sim.stats["vec"]).T
                self.vec.set_array(vec)
            if self.mbon is not None:
                mbon = np.zeros((self.sim.r_mbon.shape[-1], self.nb_frames), dtype=float)
                mbon[:, :i+1] = np.array(self.sim.stats["MBON"]).T
                self.mbon.set_array(mbon)
        else:
            self.direction.set_array(self.sim.r_direction)
            self.sigmoid_neuron.set_array(self.sim.r_sigmoid_neuron)
            self.memory.set_array(self.sim.r_memory)
            self.steering.set_array(self.sim.r_steering)
            self.steering_diff.set_array(self.sim.r_steering_diff)

        x, y = np.array(self.sim.stats["xyz"])[..., :2].T
        self.line_c.set_data(y, x)
        self.pos.set_offsets(np.array([y[-1], x[-1]]))

        vert, codes = self._marker
        vertices = R.from_euler('Z', -self.sim.agent.ori.as_euler('ZYX', degrees=True)[0], degrees=True).apply(vert)
        self.pos.set_paths((Path(vertices[:, :2], codes),))

        return time

    @property
    def sim(self):
        """
        The path integration simulation instance.

        Returns
        -------
        PathIntegrationSimulation
        """
        return self._sim

    @property
    def omm(self):
        """
        The collection of the DRA ommatidia in the figure.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[0]

    @property
    def direction(self):
        """
        The history of the CPU1 response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[1]


    @property
    def sigmoid_neuron(self):
        """
        The history of the CPU1 response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[2]


    @property
    def memory(self):
        """
        The history of the CPU1 response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[3]


    @property
    def steering(self):
        """
        The history of the CPU1 response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[4]

    @property
    def steering_diff(self):
        """
        The history of the CPU1 response in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        return self._lines[5]

    @property
    def line_c(self):
        """
        The line representing the ongoing path of the agent in the figure.

        Returns
        -------
        matplotlib.lines.Line2D
        """
        return self._lines[6]

    @property
    def line_b(self):
        """
        The line representing the finished path of the agent in the figure.

        Returns
        -------
        matplotlib.lines.Line2D
        """
        return self._lines[7]

    @property
    def pos(self):
        """
        The current position of agent in the figure.

        Returns
        -------
        matplotlib.collections.PathCollection
        """
        return self._lines[8]

    @property
    def vec(self):
        """
        The history of the Vector neurons in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        if len(self._lines) > 9:
            return self._lines[9]
        else:
            return None

    @property
    def mbon(self):
        """
        The history of the Vector neurons in the figure.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        if len(self._lines) > 10:
            return self._lines[10]
        else:
            return None