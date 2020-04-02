import scipy.integrate
import inspect
from copy import copy


class Model:
    """Class for interacting with models in in-silico CBC experiments.
    Ducktyped to act like a dict. The class requires two variables to
    be set:

        model : function
            function with signature model(x, t, pars)

            Returns the right-hand side of a system, given state x,
            time t, and parameter vector pars. pars is a dict, where
            pars[i] gives the parameter value of parameter i.

        parvec : iterable of strings 
            Defines the name of each of the parameters in the pars
            dictionary. Eg. for a model with two free parameters Iapp
            and gca, we would have parvec = ["Iapp", "gca"]

    Optional variables can be modified:

        _parameter_type : function
            By default, parameters are cast to floats. The
            self._parameter_type function is used for casting. This
            can be set to, eg. complex, int, etc., if different
            parameter types are desired.

        solver : function
            By default, the scipy.integrate.solve_ivp solver is used
            for running models. Alternative solvers can be provided
            with self.solver. The solver must have arguments (func,
            *args, **kwargs), where func is a function representing a
            system RHS, of signature RHS(t,x).

    Any parameters defined in parvec must be initialised with a value
    before running the model. All data can be provided either in the
    constructor call, or using the square bracket __setitem__ syntax.
    """

    __hash__ = None
    __str__ = None

    def __init__(self, **kw):
        self.__dict__ = kw
        if not "_parameter_type" in self.__dict__.keys():
            self._parameter_type = float

    def run_model(self, *args, **kwargs):
        """Simulate the model. Raises the following exceptions:
            AttributeError : self.model does not exist

            ValueError : self.model is not a callable function

            ValueError : self.model does not take exactly three
            arguments

            Passes up any exceptions raised by the solver and by
            construct_param_vector.

        Any args, kwargs provided are passed to the solver. All
        arguments required by the solver, besides the function itself,
        must be passed in this way."""
        # Check model has been defined
        if not "model" in self.__dict__.keys():
            raise AttributeError("No model has been defined")
        # Check it's a callable function
        if not callable(self.model):
            raise ValueError("Model is not callable function")
        # Check number of arguments of model is correct
        n_args = len(inspect.signature(self.model).parameters)
        if n_args != 3:
            raise ValueError(
                "Model must take exactly three arguments (state, time, parameters)"
            )
        # Parameter checks are done in the parameter vector construction step
        params = self.construct_param_vector()
        if not "solver" in self.__dict__:
            solver = scipy.integrate.solve_ivp
        else:
            solver = self.solver

        def binded_model(t, x): return self.model(x, t, params)
        self.solution = solver(binded_model, *args, **kwargs)
        return self.solution

    def construct_param_vector(self):
        """Collect together the parameter values defined in parvec,
        and load them into a dictionary. Cast them into the type
        returned by self._parameter_type.

        Passes up any exceptions raised by
        _construct_typed_param_vector.

        Returns a dict, mapping each string in parvec to an associated
        typed value."""
        # Allows the code to be changed to handle complex parameters, if desired
        return self._construct_typed_param_vector(self._parameter_type)

    def _construct_typed_param_vector(self, casting_func):
        """Collect together the parameter values defined in parvec,
        and load them into a dictionary. Cast them into the type
        returned by self._parameter_type. Raises the following
        exceptions:

            AttributeError : parvec has not been defined

            AttributeError : a parameter in parvec has not been
            initialised with a value

            ValueError : a parameter cannot be cast to the desired
            type

        Returns a dict, mapping each string in parvec to an associated
        typed value."""
        # Check the parameter vector has been defined
        if not "parvec" in self.__dict__.keys():
            raise AttributeError(
                "Parameter vector parvec must be defined before running a model"
            )
        for key in self.parvec:
            # Check each value in the parameter vector has been initialised
            if not key in self.__dict__.keys():
                raise AttributeError(
                    "Parameter {0} defined but not initialised".format(key)
                )

            # Check each value can be cast to the appropriate type
            value = self.__dict__[key]
            try:
                casting_func(value)
            except ValueError:
                raise ValueError(
                    "Parameter {0} (value {1}) could not be cast to a {2}".format(
                        key, value, casting_func.__name__
                    )
                )

        # Return a dict of parameter values, ordered according to 'parvec'
        return {key: casting_func(self.__dict__[key]) for key in self.parvec}

    """
    Methods required for dict ducktyping.
    Taken from PyDSTool code.
    """

    def _infostr(self, verbose=1, attributeTitle="Model", ignore_underscored=False):
        # removed offset=0 from arg list
        if len(self.__dict__) > 0:
            res = "%s (" % attributeTitle
            for k, v in self.__dict__.items():
                if k[0] == "_" and ignore_underscored:
                    continue
                if verbose == 0:
                    # don't resolve any deeper
                    if hasattr(v, "name"):
                        name = " " + v.name
                    else:
                        name = ""
                    istr = str(type(v)) + name
                else:
                    try:
                        istr = v._infostr(verbose - 1)  # , offset+2)
                    except AttributeError:
                        istr = str(v)
                res += "\n%s%s = %s," % (" ", k, istr)
                # was " "*offset
            # skip last comma
            res = res[:-1] + "\n)"
            return res
        else:
            return "No %s defined" % attributeTitle

    def __repr__(self):
        return self._infostr()

    def info(self):
        print(self._infostr())

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

    def update(self, d):
        self.__dict__.update(d)

    def clear(self):
        self.__dict__.clear()

    def get(self, k, d=None):
        return self.__dict__.get(k, d)

    def has_key(self, k):
        return k in self.__dict__

    def pop(self, k, d=None):
        return self.__dict__.pop(k, d)

    def popitem(self):
        raise NotImplementedError

    def __contains__(self, v):
        return self.__dict__.__contains__(v)

    def fromkeys(self, S, v=None):
        raise NotImplementedError

    def setdefault(self, d):
        raise NotImplementedError

    def __delitem__(self, k):
        del self.__dict__[k]

    def copy(self):
        return copy(self)

    def __cmp__(self, other):
        return self.__dict__ == other

    def __eq__(self, other):
        return self.__dict__ == other

    def __ne__(self, other):
        return self.__dict__ != other

    def __gt__(self, other):
        return self.__dict__ > other

    def __ge__(self, other):
        return self.__dict__ >= other

    def __lt__(self, other):
        return self.__dict__ < other

    def __le__(self, other):
        return self.__dict__ <= other

    def __len__(self):
        return len(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__)

    def __add__(self, other):
        d = self.__dict__.copy()
        d.update(other.__dict__)
        return args(**d)
