#!/usr/bin/env python3

from model_class import Model
from controller import Controller
import matplotlib.pyplot as plt
import math


def fitzhugh_nagumo_neuron(x, t, pars):
    """Defines the RHS of the FH model"""
    v, w = x  # Unpack state
    v_dot = v - (v**3)/3 - w + pars["I"]
    w_dot = 0.08*(v + 0.7 - 0.8*w)
    return [v_dot, w_dot]

def control_target(t):
    return [math.cos(t), math.sin(t)]

def main():
    controller = Controller()
    controller["type"] = "state"
    controller["target"] = control_target
    controller["gains"] = [1, 1]
    controller["B_matrix"] = [1, 0]
    
    model = Model()
    model["model"] = fitzhugh_nagumo_neuron
    model["parvec"] = ["I"]
    model["I"] = 1
    model["controller"] = controller
    model["openloop"] = False

    fig, ax = plt.subplots()
    solution = model.run_model([0, 100], [-1, -1])  # t_span, initial_contition
    ax.plot(solution.t, solution.y[0])
    plt.show()


if __name__ == '__main__':
    main()
