#!/usr/bin/env python3

from model_class import Model
import matplotlib.pyplot as plt


def fitzhugh_nagumo_neuron(x, t, pars):
    """Defines the RHS of the FH model"""
    v, w = x  # Unpack state
    v_dot = v - (v**3)/3 - w + pars["I"]
    w_dot = 0.08*(v + 0.7 - 0.8*w)
    return [v_dot, w_dot]


def main():
    model = Model()
    model["model"] = fitzhugh_nagumo_neuron
    model["parvec"] = ["I"]
    model["I"] = 1
    """
    Alternatively, we could write
      model = model_class.Model(model=fitzhugh_nagumo_neuron, parvec=["I"], "I"=1)
    """

    fig, ax = plt.subplots()
    solution = model.run_model([0, 100], [-1, -1])  # t_span, initial_contition
    ax.plot(solution.t, solution.y[0])
    plt.show()


if __name__ == '__main__':
    main()
