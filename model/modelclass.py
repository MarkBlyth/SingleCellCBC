import scipy.integrate
from .controller import Controller
import numpy as np


class Model:
    """
    Class for easily controlling models in in-silico CBC experiments.
    The class requires two variables to be set:

        model : function
            function with signature model(x, t, **pars)

            Returns the right-hand side of a system, given state x,
            time t, and parameters pars. Parameters **pars are keyword
            arguments for any model parameters.

        parvec : iterable of strings
            Defines the name of each of the parameters in the pars
            dictionary. Eg. for a model with two free parameters Iapp
            and gca, we would have parvec = ["Iapp", "gca"]

    Optional variables can be modified using the relevant setter
    methods:

        Model.controller : Controller object
            Controller object for applying a control scheme to the
            specified model.

        Model.uncontrolled : bool
            Boolean specifying whether a controller should be used. If
            true or unspecified, the model is run without a
            controller, in open loop. If false and a controller is
            defined, the controller is used instead. An exception is
            raised if uncontrolled is false and no controller has been
            specified.

    Any parameters defined in parvec must be initialised with a value
    before running the model. All data can be provided either in the
    constructor call, or using the square bracket __setitem__ syntax.
    """

    def __init__(self, model, parvec, uncontrolled=True, controller=None):
        """
        TODO docstring

            ValueError : self.model is not a callable function

            ValueError : self.model does not take exactly three
            arguments

        """
        # Check model is a callable function
        if not callable(model):
            raise ValueError("Model is not callable function")
        self.model = model
        self.parvec = parvec
        self.uncontrolled = uncontrolled
        self.controller = controller
        self.solution = None

    def run_model(self, t_span, initial_conds=None, **kwargs):
        """
        Run the model system, using a controller where appropriate.
        Initial conditions can be omitted if the system has been run
        before, in which case they will default to the final state of
        the previous run. Any parameters from parvec must be specified
        in kwargs. Any arguments for the ODE solver, eg. t_eval, rtol,
        etc., can be passed as kwargs too.

            t_span : 2-tuple
                Lower and upper bounds on the simulation's time
                variable.

            initial_conds : 1-by-n float array
                Initial conditions for the ODE solver. If None, the
                last state of the last ODE solver run are used. If no
                previous run has taken place and initial_conds are
                still None, an exception is raised.

            kwargs:
                Any parameters defined in parvec must have their
                values defined here. Any extra keyword arguments are
                passed to the ODE solver.

        Raises the following:

            AttributeError : uncontrolled=False, meaning a controlled
            system is required, but no controller has been specified

            TypeError : self.controller is not a valid Controller
            object

            AttributeError : a parameter in parvec has not been
            initialised with a value

            Passes up any exceptions raised by the ODE solver.

        Returns a bunch object, as returned by the ODE solver.
        """
        # Check the parameter vector has been defined
        for key in self.parvec:
            if key not in kwargs:
                raise AttributeError(
                    "Parameter {0} defined, but no value provided".format(key)
                )
        # If no initial conditions have been set, use the final state
        # from the previous run
        if initial_conds is None:
            if self.solution is None:
                raise ValueError(
                    "No previous runs so initial conditions must be provided.")
            initial_conds = self.solution.y[-1]

        # Build a dict of kwargs for the model parameters and the solver kwargs
        param_kwargs = {key: kwargs[key] for key in self.parvec}
        solver_kwargs = {key: kwargs[key]
                         for key in kwargs if key not in self.parvec}
        # Check if we should be controlling the system
        if (
            not self.uncontrolled
            and self.controller is None
        ):
            raise AttributeError(
                "Closed loop system has been requested, but no controller has been specified. Specify a controller using set_controller."
            )
        if not self.uncontrolled:
            # Get controller and bind it to model
            control = self.controller.get_controller()

            def binded_model(t, x):
                # Bind params, and add a control action
                x_dot = np.array(self.model(x, t, **param_kwargs))
                u = control(x_dot, x, t)
                return u.reshape(x_dot.shape)
        else:
            # No control requested, so run uncontrolled model
            def binded_model(t, x): return self.model(x, t, **param_kwargs)

        self.solution = scipy.integrate.solve_ivp(
            binded_model, t_span, initial_conds, **solver_kwargs)
        return self.solution

    def set_uncontrolled_flag(self, new_val):
        """
        Set the Model.uncontrolled flag. This flag is True if the
        system should be run without a controller, and False if the
        system should be run with a controller. If True, a suitable
        Controller object must be set, using set_controller.

            new_val : bool
                The new value of the Model.uncontrolled flag.

        Raises TypeError if new_val is not a bool.
        """
        if not isinstance(new_val, bool):
            raise TypeError("new_val must be a bool")
        self.uncontrolled = new_val

    def set_controller(self, controller):
        """
        Set the controller object. This should be the instance of
        Controller that we wish to use for controlling the system.

            controller : Controller obj
                Controller that we wish to use.

        Raises TypeError if controller is not a valid Controller
        object.
        """
        # Ensure we have a valid control object
        if not isinstance(self.controller, Controller):
            raise TypeError(
                "Specified controller is not a valid Controller object")
        self.controller = controller
