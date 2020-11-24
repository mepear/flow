"""Script containing the TraCI vehicle kernel class."""
import traceback

# TODO: flow.core.kenel.person.TraCIPerson

from flow.core.kernel.person import KernelPerson
import traci.constants as tc
from traci.exceptions import FatalTraCIError, TraCIException
import numpy as np
import collections
import warnings
from bisect import bisect_left
import itertools
from copy import deepcopy

# colors for vehicles
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
STEPS = 10
rdelta = 255 / STEPS
# smoothly go from red to green as the speed increases
color_bins = [[int(255 - rdelta * i), int(rdelta * i), 0] for i in
              range(STEPS + 1)]


class TraCIPerson(KernelPerson):
    """Flow kernel for the TraCI API.

    Extends flow.core.kernel.person.base.KernelVehicle
    """

    def __init__(self,
                 master_kernel,
                 sim_params):
        """See parent class."""
        KernelPerson.__init__(self, master_kernel, sim_params)

        self.__ids = []  # ids of all persons

        # persons: Key = Person ID, Value = Stage describing the person
        # Ordered dictionary used to keep neural net inputs in order
        self.__persons = collections.OrderedDict()

        # reservations
        self.__reservations = []

        # current number of persons in the network
        self.num_persons = 0

        # whether or not to automatically color vehicles
        self._force_color_update = False

    def initialize(self, persons):
        """Initialize persons state information.

        This is responsible for collecting person type information from the
        PersonParams object and placing them within the Persons kernel.

        Parameters
        ----------
        persons : flow.core.params.PersonParams
            initial person parameter information, including the types of
            individual persons and their initial speeds
        """
        self.num_persons = 0
        self.__persons = collections.OrderedDict()
        self.__ids = []
        self.__reservations = []

    def update(self, reset):
        """See parent class.

        The following actions are performed:

        * The state of all persons is modified to match their state at the
          current time step. This includes states specified by sumo, and states
          explicitly defined by flow, e.g. "num_arrived".
        * If persons exit the network, they are removed from the persons
          class, and newly departed persons are introduced to the class.

        Parameters
        ----------
        reset : bool
            specifies whether the simulator was reset in the last simulation
            step
        """

        self.__ids = self.kernel_api.person.getIDList()
        self.num_persons = len(self.__ids)
        self.__persons = collections.OrderedDict()
        for per_id in self.__ids:
            self.__persons[per_id] = {}
            self.__persons[per_id]['stage'] = self.kernel_api.person.getStage(per_id)
            self.__persons[per_id]['lane_position'] = self.kernel_api.person.getLanePosition(per_id)
            self.__persons[per_id]['lane_id'] = self.kernel_api.person.getLaneID(per_id)
            self.__persons[per_id]['position'] = self.kernel_api.person.getPosition(per_id)
        self.__reservations = self.kernel_api.person.getTaxiReservations(0)

    def add(self, per_id, type_id, edge, pos):
        self.kernel_api.person.add(per_id, edge, pos, typeID=type_id)

    def remove(self, per_id):
        self.kernel_api.person.removeStages(per_id)
    
    def color(self, per_id):
        """See parent class.

        This does not pass the last term (i.e. transparency).
        """
        r, g, b, t = self.kernel_api.person.getColor(per_id)
        return r, g, b

    def set_color(self, per_id, color):
        """See parent class.

        The last term for sumo (transparency) is set to 255.
        """
        r, g, b = color
        self.kernel_api.person.setColor(personID=per_id, color=(r, g, b, 255))

    def get_type(self, per_id):
        """Return the type of the person of per_id."""
        return 'request'

    def get_ids(self):
        """See parent class."""
        return self.__ids

    def get_waiting_ids(self):
        """See parent class."""
        ret_ids = []
        for per_id in self.__ids:
            if self.__persons[per_id]['stage'].description == 'waiting for taxi':
                ret_ids.append(per_id)
        return ret_ids
    
    def get_driving_ids(self):
        """See parent class."""
        ret_ids = []
        for per_id in self.__ids:
            if self.__persons[per_id]['stage'].description == 'driving':
                ret_ids.append(per_id)
        return ret_ids

    def get_reservations(self):
        """See parent class."""
        return self.__reservations

    def get_position(self, per_id):
        """See parent class."""
        if isinstance(per_id, (list, np.ndarray)):
            return [self.get_position(perID) for perID in per_id]
        return self.__persons[per_id]['lane_position']

    def get_lane(self, per_id):
        """See parent class."""
        if isinstance(per_id, (list, np.ndarray)):
            return [self.get_edge(perID) for perID in per_id]
        return self.__persons[per_id]['lane_id']

    def get_2d_position(self, per_id):
        """See parent class."""
        return self.__persons[per_id]['position']