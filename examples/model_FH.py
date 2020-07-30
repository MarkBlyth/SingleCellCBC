#!/usr/bin/env python3

from model import Model
import matplotlib.pyplot as plt


def fitzhugh_nagumo_neuron(x, t, I):
    """Defines the RHS of the FH model"""
    v, w = x  # Unpack state
    v_dot = v - (v ** 3) / 3 - w + I
    w_dot = 0.08 * (v + 0.7 - 0.8 * w)
    return [v_dot, w_dot]


def main():
    model = Model(model=fitzhugh_nagumo_neuron, parvec=["I"])

    fig, ax = plt.subplots()
    solution = model.run_model(
        [0, 100], [-1, -1], I=1, rtol=1e-6
    )  # t_span, initial_contition, pars + solver kwargs
    ax.plot(solution.t, solution.y[0])
    plt.show()


if __name__ == "__main__":
    main()
