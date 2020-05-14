#!/usr/bin/env python3

from model.modelclass import Model
from model.controller import Controller
import matplotlib.pyplot as plt
import numpy as np
import math


def fitzhugh_nagumo_neuron(x, t, pars):
    """Defines the RHS of the FH model"""
    v, w = x  # Unpack state
    v_dot = v - (v**3)/3 - w + pars["I"]
    w_dot = 0.08*(v + 0.7 - 0.8*w)
    return [v_dot, w_dot]

def control_target(t):
    return math.sin(t)

def euler_integrator(f, timerange, x0):
    """PID control only works with monotonously increasing time. Good ODE
    solvers evaluate the derivative at various points on each step,
    which means t won't monotonously increase over successive calls to
    the RHS. Instead, we need to use a different solver where this
    doesn't happen. This provides a quick and dirty Euler solver.
    """
    ret = [np.array(x0)]
    n_ts = int((timerange[1]-timerange[0])/0.01)
    ts = np.linspace(timerange[0], timerange[1], n_ts)
    for t in ts:
        ret.append(ret[-1] + np.array(f(t,ret[-1]))*0.01)
    return ts,np.array(ret[1:])

def main():
    controller = Controller()
    controller["type"] = "PID"
    controller["target"] = control_target
    controller["B_matrix"] = [1, 0]
    controller["C_matrix"] = np.array([[1,0]])
    controller.kp = 10
    controller.ki = 10
    controller.kd = 0
    
    model = Model()
    model["model"] = fitzhugh_nagumo_neuron
    model["parvec"] = ["I"]
    model["I"] = 1
    model["controller"] = controller
    model["openloop"] = False
    model.solver = euler_integrator

    solution = model.run_model([0, 100], [-1, -1])  # t_span, initial_contition
    ts = solution[0]
    xs = solution[1]
    fig, ax = plt.subplots()
    ax.plot(ts,xs.T[0])
    plt.show()


if __name__ == '__main__':
    main()
