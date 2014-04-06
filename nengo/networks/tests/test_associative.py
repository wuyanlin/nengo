import logging

import numpy as np
import pytest

import nengo
from nengo.networks.associative import AutoAssociative
from nengo.utils.distributions import UniformHypersphere

logger = logging.getLogger(__name__)


def test_autoassociative_intercepts():
    """Test the _calculate_intercept helper method in AutoAssociative"""

    n = 500
    dimensions = range(2, 20)
    dp = 0.05
    np.random.seed(1337)

    for d in dimensions:
        # Quick way to get n^2 sample points of np.dot(u, v) for random (u, v).
        actual = np.dot(
            UniformHypersphere(d, surface=True).sample(n),
            UniformHypersphere(d, surface=True).sample(n).T).flatten()

        # For each possible probability, check that the given intercept c does
        # in fact give np.dot(u, v) >= c for random (u, v) with probability p.
        for p in np.arange(dp, 1+dp, dp):
            intercept = AutoAssociative._calculate_intercept(d, p)
            actual_p = len(actual[actual >= intercept]) / float(len(actual))
            assert np.allclose(p, actual_p, atol=0.05)  # wow, such win
            assert np.allclose(  # test that the inverse function works
                1.0 / AutoAssociative.calculate_max_capacity(d, intercept), p, atol=0.1)


def test_autoaossiciatve(Simulator, nl_nodirect):
    n = 1000
    period = 1  # duration to show each (key, value) pair

    # Precomputed orthonormal matrix
    # (because we can't assume we have scipy.linalg.sqrtm)
    keys = np.asarray([
        [0.48745907, 0.58118766, 0.54751107, 0.35079601, 0.04217054],
        [0.57993287, 0.10103264, -0.62474875, 0.06303814, -0.5091026],
        [0.33175118, -0.268611, 0.46456557, -0.70109974, -0.33230599],
        [-0.08443313, -0.54001621, 0.30084936, 0.60231365, -0.49795728],
        [-0.55576409, 0.53688354, 0.05990926, -0.13654951, -0.61696633]])
    num_items = len(keys)

    # Verify that it's orthonormal
    assert np.allclose(np.dot(keys, keys.T), np.eye(num_items))
    d_key = len(keys[0])
    d_value = 3
    learn_time = period * num_items

    # Randomly generate some values
    np.random.seed(9001)
    values = UniformHypersphere(d_value).sample(num_items)

    def make_autoaossiciatve(**kwargs):
        model = AutoAssociative(
            nl_nodirect(n), 2*d_key, d_key, d_value, **kwargs)
        with model:
            key = nengo.Node(output=lambda t: keys[(t/period) % len(keys)])
            value = nengo.Node(output=lambda t: values[t/period]
                               if t < learn_time else np.zeros(d_value))
            learning = nengo.Node(output=lambda t: t < period * num_items)
            nengo.Connection(key, model.key)
            nengo.Connection(value, model.value)
            nengo.Connection(learning, model.learning)
            output_p = nengo.Probe(model.output, filter=0.03)
            has_key_p = nengo.Probe(model.has_key, filter=0.03)
            dopamine_p = nengo.Probe(model.dopamine, filter=0.03)
        return model, output_p, has_key_p, dopamine_p

    # Iterates through all of the (key, value) pairs twice. The second time,
    # learning is turned off.
    model, output_p, has_key_p, dopamine_p = make_autoaossiciatve(seed=123)
    sim = nengo.Simulator(model)
    sim.run(2 * learn_time)

    # Run it again but without Voja's rule.
    stupid_probes = make_autoaossiciatve(seed=123, voja_disable=True)
    stupid_model, stupid_output_p = stupid_probes[:2]
    stupid_sim = nengo.Simulator(stupid_model)
    stupid_sim.run(2 * learn_time)

    t = sim.trange()
    for i, value in enumerate(values):
        st = i * period + learn_time
        t_test = (st <= t) & (t <= st + period)

        # Check that each value is in the right ball park during recall
        output = np.average(sim.data[output_p][t_test], axis=0)
        has_key = np.average(sim.data[has_key_p][t_test], axis=0)
        dopamine = np.average(sim.data[dopamine_p][t_test], axis=0)
        assert np.allclose(value, output, atol=0.05)
        assert np.allclose([1], has_key, atol=0.02)
        assert np.allclose(dopamine, 0, atol=0.05)

        # Check that these values are represented better than without Voja's
        # rule, except for the most recent one since there's effectively no
        # difference for that value (and may even be worse if most of the
        # encoders have been used up).
        if i != len(values) - 1:
            stupid_output = np.average(
                stupid_sim.data[stupid_output_p][t_test], axis=0)
            assert (np.linalg.norm(output - value) + 0.01 <
                    np.linalg.norm(stupid_output - value))


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
