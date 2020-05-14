import numpy as np
from .pid import _PID
import scipy.misc


class Controller:
    """Class for constructing controllers. Methods are used to set up a
    controller. The result is a function that gives the appropriate
    control action at time t. This function can be used in addition to
    a system to provide a model of a closed loop process. Two
    variables must be set:

        type : str
            String describing which control strategy to apply.

        target : function 
            Function with signature target(t). Returns the target
            vector at time t that the controller seeks to stabilise.

        B_matrix : ndarray
            Vector denoting which state variables the control action
            acts on. Must contain the same number of elements as the
            state vector.

    Several variables are required in some contexts, and optional in
    others. For PID controllers, the following must be set:

        kp : float
            Proportional feedback gain for a PID controller. Must be
            set if proportional control is desired. Can be left unset
            otherwise.

        ki : float
            Integral feedback gain for a PID controller. Must be set
            if integral control is desired. Can be left unset
            otherwise.

        kd : float
            Derivative feedback gain for a PID controller. Must be set
            if derivative control is desired. Can be left unset
            otherwise.

        C_matrix: ndarray
            Matrix mapping a system state to an observed output. Must
            multiply with the system state vector to produce a scalar
            value.

    For state feedback control, with state of dimension n, all of the
    following must be set:

        gains : ndarray
            Control gains. gain[i] represents the gain applied to the
            i'th state variable. Must be exactly one gain per state
            variable.

    Integral control for PID controllers produces a system that is not
    suitable for traditional numerical methods, and therefore requires
    a custom integrator. This should be avoided.

    """

    def __init__(self, **kw):
        self.__dict__ = kw
        PID_dict = {x: self._integral_controller for x in "I PI ID PID".split()}
        PD_dict = {x: self._derivative_controller for x in "P D PD".split()}
        self._controller_types = {
            "state": self._state_feedback_controller,
            **PID_dict,
            **PD_dict,
        }

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
        if not "B_matrix" in self.__dict__:
            raise AttributeError("No B matrix defined")
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

        # Force into correct shapes and check sizes match
        gains = np.array(self.gains).reshape((1, -1))
        B_matrix = np.array(self.B_matrix).reshape((-1, 1))
        if not B_matrix.shape[0] == gains.shape[1]:
            raise ValueError(
                "B matrix and gains must contain the same number of elements"
            )

        # Build control function
        def controlled_RHS(RHS, x, t):
            """Return function for giving the controlled system RHS,
            for uncontrolled RHS, state x, time t.

                RHS : ndarray
                    The derivative of the system, evaluated at state
                    x, time t

                x : ndarray
                    Vector of floats representing the system state at
                    the current instant

                t : float Current time

            Raises the following exceptions:

                Raises ValueError if the dimension of the state
                doesn't correctly correspond to that of the gains

            Returns the new, controlled RHS of the ODE system, with a
            full state feedback control strategy applied.
            """
            # Check state dimensions match
            state = np.array(x).reshape((-1, 1))
            if not gains.T.shape == state.shape:
                raise ValueError(
                    "State vector, B matrix, and gains must contain the same number of elements"
                )
            error = state - self._control_target(t)
            return state - B_matrix * np.dot(gains, error)

        return controlled_RHS

    def _integral_controller(self):
        # Existence and correctness of self.type is checked for in the calling function.
        # Ensure all the gains have been defined
        if "P" in self.type and not "kp" in self.__dict__:
            raise AttributeError("No proportional feedback gain set")
        if "I" in self.type and not "ki" in self.__dict__:
            raise AttributeError("No integral feedback gain set")
        if "D" in self.type and not "kd" in self.__dict__:
            raise AttributeError("No derivative feedback gain set")
        if not "C_matrix" in self.__dict__:
            raise AttributeError("No C matrix defined")
        kp = 0 if not "P" in self.type else self.kp
        ki = 0 if not "I" in self.type else self.ki
        kd = 0 if not "D" in self.type else self.kd
        return _PID(kp, ki, kd, self.B_matrix, self.C_matrix, self._control_target)

    def _derivative_controller(self):
        # Existence and correctness of B_matrix and self.type is
        # checked for in the calling function. Ensure all the gains
        # have been defined
        if "P" in self.type and not "kp" in self.__dict__:
            raise AttributeError("No proportional feedback gain set")
        if "D" in self.type and not "kd" in self.__dict__:
            raise AttributeError("No derivative feedback gain set")
        if not "C_matrix" in self.__dict__:
            raise AttributeError("No C matrix defined")
        kp = 0 if not "P" in self.type else self.kp
        kd = 0 if not "D" in self.type else self.kd

        # See if we have the control target derivative
        # If not, construct it
        if "derivative" in self.__dict__:
            derivative = self.target_derivative
        else:
            derivative = lambda t: scipy.misc.derivative(self._control_target, t)

        B_matrix = np.array(self.B_matrix).reshape((-1, 1))
        # Remove to allow multidimensional u
        C_matrix = np.array(self.C_matrix).reshape((1, -1))
        # Construct the derivative controller
        BC_mat = np.dot(B_matrix, C_matrix)
        derivative_control_mat = np.linalg.inv(np.eye(BC_mat.shape[0]) - kd * BC_mat)

        def controlled_RHS(RHS, x, t):
            """Return function for giving the controlled system RHS,
            for uncontrolled RHS, state x, time t.

                RHS : ndarray
                    The derivative of the system, evaluated at state
                    x, time t

                x : ndarray
                    Vector of floats representing the system state at
                    the current instant

                t : float Current time

            Raises the following exceptions:

                Raises ValueError if the dimension of the state
                doesn't correctly correspond to that of the gains

            Returns the new, controlled RHS of the ODE system, with a
            PD feedback control strategy applied.
            """
            state = np.array(x).reshape((-1, 1))
            if not state.shape == B_matrix.shape:
                raise ValueError(
                    "B matrix must contain the same number of elements as the system state"
                )
            if not C_matrix.T.shape == state.shape:
                raise ValueError(
                    "C matrix must contain the same number of elements as the system state"
                )
            target_derivative = derivative(t)
            control_scalar = (
                kp * (np.dot(C_matrix, state) - self._control_target(t))
                - kd * target_derivative
            )
            uncorrected_RHS = np.array(RHS).reshape((-1, 1)) + B_matrix * control_scalar
            return np.dot(derivative_control_mat, uncorrected_RHS)

        return controlled_RHS

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
