import numpy as np
from .pid import _PID
import scipy.misc


class Controller:
    """
    Class for constructing controllers. The result is a function that
    gives the controlled system RHS, for a given control strategy and
    control target. Integral control for PID controllers produces a
    system that is not suitable for standard numerical methods, and
    therefore requires a custom integrator. This should be avoided.

    Controlled systems are of the form
        xdot = f(x) + C u(Bx, t)
    for PID control. C decides which state variables the controller is
    able to act on, and B decides how to map the system states to a
    scalar value; the controller seeks to drive this scalar value to
    match the control target.
    """

    def __init__(self, controller_type, B_matrix, control_target, **kw):
        """
        Construct a controller class. Once constructed, controller
        functions can be produced for a given control target, using
        the get_controller function.

            controller_type : str
                String describing which control strategy to apply. Must be
                one of {P, I, D, PI, PD, PID, ID, I, state}.

            B_matrix : ndarray
                Vector denoting which state variables the control action
                acts on. Must contain the same number of elements as the
                state vector.

            control_target : function
                Function of signature control_target(t). Returns the
                appropriately dimensioned control target for the
                system at time t.

            **kw :
                Any additional arguments for the chosen controller
                type. Several variables are required in some contexts,
                and optional in others. For PID controllers, the
                following must be set:

                    kp : float
                        Proportional feedback gain for a PID
                        controller. Must be set if proportional
                        control is desired. Can be left unset
                        otherwise.

                    ki : float
                        Integral feedback gain for a PID controller.
                        Must be set if integral control is desired.
                        Can be left unset otherwise.

                    kd : float
                        Derivative feedback gain for a PID controller.
                        Must be set if derivative control is desired.
                        Can be left unset otherwise.

                    C_matrix: ndarray
                        Matrix mapping a system state to an observed
                        output. Must multiply with the system state
                        vector to produce a scalar value; the
                        controller then tries to make this scalar
                        value track the control target.

                For state feedback control, with state of dimension n, the
                following must be set:

                    gains : ndarray
                        Control gains. gain[i] represents the gain
                        applied to the i'th state variable. Must be
                        exactly one gain per state variable.

        For derivative control, the time derivative of the control
        target can optionally be defined with the target_derivative
        kwarg.
        """
        self.__dict__ = kw
        PID_dict = {x: self._integral_controller for x in "I PI ID PID".split()}
        PD_dict = {x: self._derivative_controller for x in "P D PD".split()}
        controller_types = {
            "state": self._state_feedback_controller,
            **PID_dict,
            **PD_dict,
        }
        if controller_type not in controller_types:
            raise ValueError(
                "{0} is not a valid controller type".format(controller_type)
            )
        self._controller_builder = controller_types[controller_type]
        self.controller_type = controller_type
        self.B_matrix = B_matrix
        self.control_target = control_target

    def get_controller(self):
        """
        Build a controller, given the controller parameters set at
        init.

        Returns a function rhs(x,t), which gives a system right-hand
        side after applying the appropriate control action, for state
        x, time t, using the control strategy specified at init.
        """
        return self._controller_builder(self.control_target)

    def _state_feedback_controller(self, control_target):
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
        if "gains" not in self.__dict__:
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
            error = state - self._formatted_control_target(control_target, t)
            return state - B_matrix * np.dot(gains, error)

        return controlled_RHS

    def _integral_controller(self, control_target):
        # Ensure all the gains have been defined
        if "P" in self.controller_type and "kp" not in self.__dict__:
            raise AttributeError("No proportional feedback gain set")
        if "I" in self.controller_type and "ki" not in self.__dict__:
            raise AttributeError("No integral feedback gain set")
        if "D" in self.controller_type and "kd" not in self.__dict__:
            raise AttributeError("No derivative feedback gain set")
        if "C_matrix" not in self.__dict__:
            raise AttributeError("No C matrix defined")
        kp = 0 if "P" not in self.controller_type else self.kp
        ki = 0 if "I" not in self.controller_type else self.ki
        kd = 0 if "D" not in self.controller_type else self.kd
        return _PID(kp, ki, kd, self.B_matrix, self.C_matrix, control_target)

    def _derivative_controller(self, control_target):
        # Ensure all the gains have been defined
        if "P" in self.controller_type and "kp" not in self.__dict__:
            raise AttributeError("No proportional feedback gain set")
        if "D" in self.controller_type and "kd" not in self.__dict__:
            raise AttributeError("No derivative feedback gain set")
        if "C_matrix" not in self.__dict__:
            raise AttributeError("No C matrix defined")
        kp = 0 if "P" not in self.controller_type else self.kp
        kd = 0 if "D" not in self.controller_type else self.kd

        # See if we have the control target derivative
        # If not, construct it
        if "derivative" in self.__dict__:
            derivative = self.target_derivative
        else:

            def derivative(t):
                return scipy.misc.derivative(control_target, t)

        B_matrix = np.array(self.B_matrix).reshape((-1, 1))
        # Remove to allow multidimensional u
        C_matrix = np.array(self.C_matrix).reshape((1, -1))
        # Construct the derivative controller
        BC_mat = np.dot(B_matrix, C_matrix)
        derivative_control_mat = np.linalg.inv(
            np.eye(BC_mat.shape[0]) - kd * BC_mat)

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
                kp
                * (
                    np.dot(C_matrix, state)
                    - self._formatted_control_target(control_target, t)
                )
                - kd * target_derivative
            )
            uncorrected_RHS = np.array(RHS).reshape(
                (-1, 1)) + B_matrix * control_scalar
            corrected_RHS = np.dot(derivative_control_mat, uncorrected_RHS)
            return corrected_RHS

        return controlled_RHS

    def _formatted_control_target(self, control_target, t):
        """Evaluate the control target and reshape the result into a
        column vector.

        Returns the control target at time t, as defined by the
        function `control_target'
        """
        # Ensure control target has been defined
        return np.array(control_target(t)).reshape((-1, 1))
