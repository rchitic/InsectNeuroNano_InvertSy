from invertsy.sim.simulation import CentralPointNavigationSimulationBase

from ._helpers import col2x, row2y, yaw2ori, x2col, y2row, ori2yaw

from invertsy.__helpers import __data__, RNG
from invertsy.env import UniformSky, Sky, Seville2009, WorldBase, StaticOdour
from invertsy.agent import VisualNavigationAgent, VectorMemoryAgent, NavigatingAgent, RouteFollowingAgent, Agent
from invertsy.agent.agent import VisualProcessingAgent, PathIntegrationAgent

from invertpy.sense import CompoundEye
from invertpy.sense.polarisation import PolarisationSensor, MinimalDevicePolarisationSensor

from invertpy.brain.preprocessing import Preprocessing
from invertpy.brain.compass import PolarisationCompass
from invertpy.brain.memory import MemoryComponent
from invertpy.brain.centralcomplex import CentralComplexBase
from invertpy.brain.centralcomplex.dyememory import DyeMemoryCX
from invertpy.brain.preprocessing import MentalRotation

from scipy.spatial.transform import Rotation as R
from scipy.special import expit
import numpy as np
import loguru as lg
from time import time
from copy import copy
import os

__outb_dir__ = os.path.abspath(os.path.join(__data__, "animation", "outbounds"))
if not os.path.isdir(__outb_dir__):
    os.makedirs(__outb_dir__)

class MinimalDevicePathIntegrationSimulation(CentralPointNavigationSimulationBase):

    def __init__(self, route, zero_vector=False, *args, **kwargs):
        """
        Runs the path integration task.
        An agent equipped with a compass and the central complex is forced to follow a route and then it is asked to
        return to its initial position.

        Parameters
        ----------
        route: np.ndarray[float]
            N x 4 matrix that holds the 3D position (x, y, z) and 1D orientation (yaw) for each iteration of the route
        agent: PathIntegrationAgent, optional
            the agent that will be used for the path integration. Default is a `PathIntegrationAgent(speed=.01)`
        sky: Sky, optional
            the sky of the environment. Default is a sky with the sun to the South and 30 degrees above the horizon
        world: WorldBase, optional
            the world that creates the scenery disturbing the celestial compass or using the visual surroundings as a
            a compass. Default is None

        Other Parameters
        ----------------
        nb_iterations: int, optional
            the number of iterations to run the simulation. Default is 2.5 times the number of time-steps of the route
        name: str, optional
            the name of the simulation. Default is the name of the world or 'simulation'
        """
        if len(args) == 0:
            kwargs.setdefault("xyz", route[0, :3])
        kwargs.setdefault('nb_iterations', int(3.5 * route.shape[0]))
        super().__init__(*args, **kwargs)
        self._route = route

        if self.agent is None:
            self._agent = VectorMemoryAgent(nb_feeders=1, speed=.01, rng=self.rng, noise=self._noise)

        self._compass_sensor = None
        for sensor in self.agent.sensors:
            if isinstance(sensor, MinimalDevicePolarisationSensor):
                self._compass_sensor = sensor
        self._cx = self.agent.brain[0]

        self._foraging = True
        self._distant_point = route[-1, :3]
        self._zero_vector = zero_vector
        if isinstance(self.agent, PathIntegrationAgent) and isinstance(self.agent.central_complex, DyeMemoryCX):
            self._beta = self._agent.central_complex["memory"].beta
        else:
            self._beta = 0.0

        self.__file_data = None

    def reset(self, *args):
        """
        Initialises the logged statistics and iteration count, calibrates the eye of agent if applicable and places it
        to the beginning of the route.

        Returns
        -------
        np.ndarray[float]
            array of the 3D positions of the samples used for the calibration
        """
        super().reset()
        self._stats["direction"] = []
        self._stats["memory"] = []
        self._stats["sigmoid_neuron"] = []
        self._stats["steering"] = []
        self._stats["steering_diff"] = []

        self.__file_data = None

        self.agent.ori = R.from_euler("Z", self.route[0, 3], degrees=True)
        self._foraging = True

    def init_stats(self):
        super().init_stats()
        self._stats["direction"] = []
        self._stats["memory"] = []
        self._stats["sigmoid_neuron"] = []
        self._stats["steering"] = []
        self._stats["steering_diff"] = []

        if hasattr(self.agent, "eye"):
            self._stats["ommatidia"] = []

    def init_inbound(self):
        """
        Sets up the inbound phase.
        Changes the labels of the logged stats to their outbound equivalent and resets them for the new phase to come.
        """
        CentralPointNavigationSimulationBase.init_inbound(self)

        if self._zero_vector:
            self.agent.xyz = self.route[0, :3]
            self.agent.ori = R.from_euler("Z", self.route[0, 3], degrees=True)
            self.agent.central_complex.reset_integrator()

        if isinstance(self.agent, PathIntegrationAgent) and isinstance(self.agent.central_complex, DyeMemoryCX):
            self.agent.central_complex["memory"].beta = self._beta

        # file_path = os.path.join(__outb_dir__, f"{self.name}.npz")
        # if not os.path.exists(file_path):
        #     np.savez(file_path, **self.stats)
        #     lg.logger.info(f"Outbound stats are saved in: '{file_path}'")

    def _step(self, i):
        """
        Runs one iteration of the simulation. If the iteration is less than the maximum number of iterations in the
        route it forces the agent to follow the route, otherwise it lets the agent decide its actions.

        Parameters
        ----------
        i: int
            the iteration ID
        """
        act = True
        omm_responses = None
        if i < self._route.shape[0]:  # outbound
            x, y, z, yaw = self._route[i]
            self._agent.xyz = [x, y, z]
            self._agent.ori = R.from_euler('Z', yaw, degrees=True)
            act = False
            # for process in self.agent.preprocessing:
            #     if isinstance(process, MentalRotation):
            #         process.pref_angles[:] = np.pi

            file_path = os.path.join(__outb_dir__, f"{self.name}.npz")
            if os.path.exists(file_path) and self.__file_data is None:
                lg.logger.info(f"Loading outbound stats from: '{file_path}'")
                data = np.load(file_path, allow_pickle=True)
                self.__file_data = {
                    "ommatidia": data["ommatidia"]
                }
            if self.__file_data is not None:
                omm_responses = self.__file_data["ommatidia"][i]

            self._foraging = True
        elif i == self._route.shape[0]:
            self.init_inbound()
            self._foraging = False
            # for process in self.agent.preprocessing:
            #     if isinstance(process, MentalRotation):
            #         process.pref_angles[:] = 0.
        # elif self._foraging and self.distance_from(self.distant_point) < 0.5:
        #     self.approach_point(self.distant_point)
        # elif not self._foraging and not self._zero_vector and self.d_nest < 0.5:
        #     self.approach_point(self.central_point)
        # elif self._foraging and self.distance_from(self.distant_point) < 0.2:
        #     self._foraging = False
        #     lg.logger.debug("START PI FROM FEEDER")
        # elif not self._foraging and not self._zero_vector and self.d_nest < 0.2:
        #     self._foraging = True
        #     lg.logger.debug("START FORAGING!")

        # if self._foraging:
        #     motivation = np.array([0, 1])
        # else:
        #     motivation = np.array([1, 0])

        if hasattr(self.agent, "mushroom_body"):
            self.agent.mushroom_body.update = self._foraging

        self._agent(sky=self._sky, scene=self._world, act=act, callback=self.update_stats)

        if i > self.route.shape[0] and "replace" in self._stats:
            d_route = np.linalg.norm(self.route[:, :3] - self._agent.xyz, axis=1)
            point = np.argmin(d_route)
            if d_route[point] > 0.2:  # move for more than 20cm away from the route
                self.agent.xyz = self.route[point, :3]
                self.agent.ori = R.from_euler('Z', self.route[point, 3], degrees=True)
                self._stats["replace"].append(True)
                lg.logger.debug(" ~ REPLACE ~")
            else:
                self._stats["replace"].append(False)

    def update_stats(self, a):
        """
        Updates the logged statistics of the agent.

        Parameters
        ----------
        a: PathIntegrationAgent, NavigatingAgent
        """

        super().update_stats(a)

        cx = a.brain[0]

        self._stats["direction"].append(a.pol_sensor.r_POL.copy())
        self._stats["memory"].append(1e+07*cx.r_memory.copy())
        self._stats["sigmoid_neuron"].append(cx.r_sigmoid_neuron.copy())
        self._stats["steering"].append(cx.r_steering.copy()*1e+8)
        self._stats["steering_diff"].append(1e+8*(cx.r_steering[1].copy()-cx.r_steering[0].copy()))

        if hasattr(a, "eye"):
            if self.__file_data is not None and self._iteration < len(self.__file_data["ommatidia"]):
                self._stats["ommatidia"].append(self.__file_data["ommatidia"][self._iteration])
            else:
                self._stats["ommatidia"].append(a.eye.responses.copy())

        if hasattr(a, 'mushroom_body'):
            self._stats["PN"].append(a.mushroom_body.r_cs[0, 0].copy())
            self._stats["KC"].append(a.mushroom_body.r_kc[0, 0].copy())
            self._stats["MBON"].append(a.mushroom_body.r_mbon[0, 0].copy())
            self._stats["DAN"].append(a.mushroom_body.r_dan[0, 0].copy())
            self._stats["familiarity"].append(np.power(a.mushroom_body.familiarity[0, 0, ::2].mean(), 8) * 100)
            self._stats["capacity"].append(a.mushroom_body.free_space * 100)

    @property
    def agent(self):
        """
        The agent of the simulation.

        Returns
        -------
        PathIntegrationAgent
        """
        return self._agent

    @property
    def route(self):
        """
        N x 4 array representing the route that the agent follows before returning to its initial position.

        Returns
        -------
        np.ndarray[float]
        """
        return self._route

    @property
    def distant_point(self):
        return self._distant_point

    @property
    def compass_sensor(self):
        """
        The polarisation compass sensor.

        Returns
        -------
        PolarisationSensor
        """
        return self._compass_sensor

    @property
    def compass_model(self):
        """
        The Compass model.

        Returns
        -------
        PolarisationCompass
        """
        return self._compass_model

    @property
    def central_complex(self):
        """
        The Central Complex model.

        Returns
        -------
        CentralComplexBase
        """
        return self._cx

    @property
    def d_nest(self):
        """
        The distance between the agent and the nest.

        Returns
        -------
        float
        """
        return self.d_central_point

    @property
    def r_direction(self):
        """
        The POL responses of the compass model of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._cx.r_direction.T.flatten()

    @property
    def r_memory(self):
        """
        The memory responses of the central complex of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._cx.r_memory.T.flatten()

    @property
    def r_sigmoid_neuron(self):
        """
        The sigmoid neuron responses of the central complex of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._cx.r_sigmoid_neuron.T.flatten()

    @property
    def r_steering(self):
        """
        The steering responses of the central complex of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._cx.r_steering.T.flatten()

    @property
    def r_steering_diff(self):
        """
        The steering responses of the central complex of the agent.

        Returns
        -------
        np.ndarray[float]
        """
        return self._cx.r_steering_diff.T.flatten()