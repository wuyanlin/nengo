import logging

import numpy as np
import pytest

import nengo
from nengo.utils.compat import range
from nengo.utils.testing import Plotter

logger = logging.getLogger(__name__)


def test_multidim(Simulator, nl):
    """Test an ensemble array with multiple dimensions per ensemble"""
    dims = 3
    n_neurons = 60
    radius = 1.0

    rng = np.random.RandomState(523887)
    a = rng.uniform(low=-0.7, high=0.7, size=dims)
    b = rng.uniform(low=-0.7, high=0.7, size=dims)
    c = np.zeros(2 * dims)
    c[::2] = a
    c[1::2] = b

    model = nengo.Network(label='Multidim', seed=123)
    with model:
        model.config[nengo.Ensemble].neuron_type = nl()
        inputA = nengo.Node(a)
        inputB = nengo.Node(b)
        A = nengo.networks.EnsembleArray(n_neurons, dims, radius=radius)
        B = nengo.networks.EnsembleArray(n_neurons, dims, radius=radius)
        C = nengo.networks.EnsembleArray(n_neurons * 2, dims,
                                         ens_dimensions=2,
                                         radius=radius)
        nengo.Connection(inputA, A.input)
        nengo.Connection(inputB, B.input)
        nengo.Connection(A.output, C.input[::2])
        nengo.Connection(B.output, C.input[1::2])

        A_p = nengo.Probe(A.output, synapse=0.03)
        B_p = nengo.Probe(B.output, synapse=0.03)
        C_p = nengo.Probe(C.output, synapse=0.03)

    sim = Simulator(model)
    sim.run(1.0)

    t = sim.trange()
    with Plotter(Simulator, nl) as plt:
        def plot(sim, a, p, title=""):
            a_ref = np.tile(a, (len(t), 1))
            a_sim = sim.data[p]
            colors = ['b', 'g', 'r', 'c', 'm', 'y']
            for i in range(a_sim.shape[1]):
                plt.plot(t, a_ref[:, i], '--', color=colors[i % 6])
                plt.plot(t, a_sim[:, i], '-', color=colors[i % 6])
            plt.title(title)

        plt.subplot(131)
        plot(sim, a, A_p, title="A")
        plt.subplot(132)
        plot(sim, b, B_p, title="B")
        plt.subplot(133)
        plot(sim, c, C_p, title="C")
        plt.savefig('test_ensemble_array.test_multidim.pdf')
        plt.close()

    a_sim = sim.data[A_p][t > 0.5].mean(axis=0)
    b_sim = sim.data[B_p][t > 0.5].mean(axis=0)
    c_sim = sim.data[C_p][t > 0.5].mean(axis=0)

    rtol, atol = 0.1, 0.05
    assert np.allclose(a, a_sim, atol=atol, rtol=rtol)
    assert np.allclose(b, b_sim, atol=atol, rtol=rtol)
    assert np.allclose(c, c_sim, atol=atol, rtol=rtol)


def _mmul_transforms(A_shape, B_shape, C_dim):
    transformA = np.zeros((C_dim, A_shape[0] * A_shape[1]))
    transformB = np.zeros((C_dim, B_shape[0] * B_shape[1]))

    for i in range(A_shape[0]):
        for j in range(A_shape[1]):
            for k in range(B_shape[1]):
                tmp = (j + k * A_shape[1] + i * B_shape[0] * B_shape[1])
                transformA[tmp * 2][j + i * A_shape[1]] = 1
                transformB[tmp * 2 + 1][k + j * B_shape[1]] = 1

    return transformA, transformB


def test_arguments():
    """Make sure EnsembleArray accepts the right arguments."""
    with pytest.raises(TypeError):
        nengo.networks.EnsembleArray(nengo.LIF(10), 1, dimensions=2)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
