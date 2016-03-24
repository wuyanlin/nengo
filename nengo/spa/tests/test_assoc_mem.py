import nengo
import numpy as np
from nengo.spa import Vocabulary, Input
from nengo.spa.utils import similarity
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

    voc1.parse('A')
    voc2.parse('C+D')

    in_keys = ['0.9*A']
    out_keys = ['-C+0.5*D']

    with nengo.spa.SPA(seed=seed) as model:

        model.am = AssociativeMemory(input_vocab=voc1,
                                     output_vocab=voc2,
                                     input_keys=in_keys,
                                     output_keys=out_keys,
                                     threshold=0.0)

        model.inp = Input(am='A')
        prob = nengo.Probe(model.am.output, synapse=0.03)

    with nengo.Simulator(model) as sim:
        sim.run(0.5)
        vec_sim = sim.data[prob]

        final_vec = vec_sim[150:].mean(axis=0)
        sim = similarity(final_vec, voc2, normalize=True)[0]

        err_margin = 0.2

        # Presenting A, expected at output: -C, 0.5*D

        # Test for approximately equal values: allowed `err_margin` deviation
        # from the desired value
        np.testing.assert_allclose(sim[0], -1, atol=err_margin)
        np.testing.assert_allclose(sim[1], 0.5, atol=err_margin)
