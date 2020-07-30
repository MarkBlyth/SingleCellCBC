#!/usr/bin/env python3

from model.modelclass import Model
from model.controller import Controller
import matplotlib.pyplot as plt
import math


def fitzhugh_nagumo_neuron(x, t, I):
    """Defines the RHS of the FH model"""
    v, w = x  # Unpack state
    v_dot = v - (v ** 3) / 3 - w + I
    w_dot = 0.08 * (v + 0.7 - 0.8 * w)
    return [v_dot, w_dot]


def control_target(t):
    return math.sin(t)


def main():
    controller = Controller()
    controller["type"] = "PD"
    controller["target"] = control_target
    controller["B_matrix"] = [1, 0]
    controller["C_matrix"] = [1, 0]
    controller.kp = 10
    controller.kd = 10

    model = Model(fitzhugh_nagumo_neuron, ["I"], False, controller)

    solution = model.run_model(
        [0, 25], [-1, -1], I=1, rtol=1e-6
    )  # t_span, initial_contition, kwargs
    fig, ax = plt.subplots()
    ax.plot(solution.t, solution.y[0])
    plt.show()


if __name__ == "__main__":
    main()
