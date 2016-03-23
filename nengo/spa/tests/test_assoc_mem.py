import nengo
import numpy as np
from nengo.spa import Vocabulary, Input
from nengo.spa.assoc_mem import AssociativeMemory


def test_am_spa_interaction(Simulator, seed, rng):
    """Make sure associative memory interacts with other SPA modules."""
    D = 16
    vocab = Vocabulary(D, rng=rng)
    vocab.parse('A+B+C+D')

    D2 = int(D / 2)
    vocab2 = Vocabulary(D2, rng=rng)
    vocab2.parse('A+B+C+D')

    def input_func(t):
        return '0.49*A' if t < 0.5 else '0.79*A'

    with nengo.spa.SPA(seed=seed) as m:
        m.buf = nengo.spa.Buffer(D)
        m.input = nengo.spa.Input(buf=input_func)

        m.am = AssociativeMemory(vocab, vocab2,
                                 input_keys=['A', 'B', 'C'],
                                 output_keys=['B', 'C', 'D'],
                                 default_output_key='A',
                                 threshold=0.5,
                                 inhibitable=True,
                                 wta_output=True,
                                 threshold_output=True)

        cortical_actions = nengo.spa.Actions('am = buf')
        m.c_act = nengo.spa.Cortical(cortical_actions)

    # Check to see if model builds properly. No functionality test needed
    with Simulator(m):
        pass


def test_am_spa_keys_as_expressions(Simulator, seed, rng):
    """Provide semantic pointer expressions as input and output keys."""
    D = 64

    voc1 = Vocabulary(D, rng=rng)
    voc2 = Vocabulary(D, rng=rng)

    voc1.parse('A+B')
    voc2.parse('C+D+E')

    in_keys = ['A', '0.5*B']
    out_keys = ['-C+D', '0.3*E']

    with nengo.spa.SPA(seed=seed) as model:
        model.am = AssociativeMemory(input_vocab=voc1,
                                     output_vocab=voc2,
                                     input_keys=in_keys,
                                     output_keys=out_keys,
                                     threshold=0.0)

        model.inp = Input(am=lambda t: 'A' if t < .5 else 'B')
        prob = nengo.Probe(model.am.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(1.0)
        vec_sim = sim.data[prob]

        final_vecA = vec_sim[100:490].mean(axis=0)
        simA = np.dot(voc2.vectors, final_vecA)

        final_vecB = vec_sim[600:].mean(axis=0)
        simB = np.dot(voc2.vectors, final_vecB)

        err_margin = 0.4
        # Case 1. presenting A
        # Expected at output: -C, D

        # Test for approximately equal values
        np.testing.assert_allclose(simA[0], -simA[1], rtol=0.1)
        # Test for mapped values "close" to 1
        np.testing.assert_allclose(simA[0], -1, rtol=err_margin)
        # Test E is not affected by A (should be "close" to 0)
        np.testing.assert_allclose(np.abs(simA[2]), 0, atol=err_margin)

        # Case 2. presenting B
        # Expected output: 0.3*E

        # Test E is close to 0.3
        np.testing.assert_allclose(simB[2], 0.3, rtol=err_margin)
        # Test C and D are close to 0
        np.testing.assert_allclose(np.abs(simB[0]), 0, atol=err_margin)
        np.testing.assert_allclose(np.abs(simB[1]), 0, atol=err_margin)
