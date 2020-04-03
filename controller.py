import numpy as np


class Controller:
    def __init__(self, **kw):
        self.__dict__ = kw
        self._controller_types = {"state": self._state_feedback_controller}

    def get_controller(self):
        """Return a controller function.

            Raises AttributeError if the controller type is not
            specified

            Raises ValueError if the requested controller type is not
            valid

        Returns a function u(x,t), which gives the applied control
        action for state x, time t, using the control strategy
        specified by self.type.
        """
        if not "type" in self.__dict__:
            raise AttributeError("Controller type not specified")
        if not self.type in self._controller_types:
            raise ValueError("{0} is not a valid controller type".format(self.type))
        return self._controller_types[self.type]()

    def _state_feedback_controller(self):
        """Build a state feedback controller.

            Raises AttributeError if no controller gains have been
            specified

            Raises AttributeError if no B matrix has been specified

            Raises ValueError if the dimensions of the gains and B
            matrix are not correct

        Returns a function u(x,t), which gives the applied control
        action for state x, time t, using a full state feedback
        control strategy.
        """
        # Check relevant variables exist
        if not "gains" in self.__dict__:
            raise AttributeError("No controller gains have been defined")
        if not "B_matrix" in self.__dict__:
            raise AttributeError("No B matrix defined")

        # Force into correct shapes and check sizes match
        gains = np.array(self.gains).reshape((1, -1))
        B_matrix = np.array(self.B_matrix).reshape((-1, 1))
        if not B_matrix.shape[0] == gains.shape[1]:
            raise ValueError(
                "B matrix and gains must contain the same number of elements"
            )

        # Build control function
        def control_action(x, t):
            """Return function for giving the control action at state
            x, time t.

                x : ndarray Vector of floats representing the system
                    state at the current instant

                t : float Current time

            Raises the following exceptions:

                Raises ValueError if the dimension of the state
                doesn't correctly correspond to that of the gains

            Returns the control action u(x,t) as found by a state
            feedback control strategy.
            """
            # Check state dimensions match
            state = np.array(x).reshape((-1, 1))
            if not gains.T.shape == state.shape:
                raise ValueError(
                    "State vector, B matrix, and gains must contain the same number of elements"
                )
            error = state - self._control_target(t)
            return -B_matrix * np.dot(gains, error)

        return control_action

    def _control_target(self, t):
        """Evaluate the control target and reshape the result into a
        column vector.

            t : float Time at which to evaluate the control target
                function

            Raises AttributeError if self.target has not been set

            Passes up any exceptions from target and ndarray.reshape

        Returns the control target at time t, as defined by the
        function `target'
        """
        # Ensure control target has been defined
        if not "target" in self.__dict__:
            raise AttributeError("Control target not set")
        return np.array(self.target(t)).reshape((-1, 1))

    """
    Some methods for dict-like interaction
    """

    def values(self):
        return list(self.__dict__.values())

    def keys(self):
        return list(self.__dict__.keys())

    def items(self):
        return list(self.__dict__.items())

    def itervalues(self):
        return iter(self.__dict__.values())

    def iterkeys(self):
        return iter(self.__dict__.keys())

    def iteritems(self):
        return iter(self.__dict__.items())

    def __getitem__(self, k):
        return self.__dict__[k]

    def __setitem__(self, k, v):
        self.__dict__.__setitem__(k, v)

    def get(self, k, d=None):
        return self.__dict__.get(k, d)

    def has_key(self, k):
        return k in self.__dict__

    def __contains__(self, v):
        return self.__dict__.__contains__(v)
