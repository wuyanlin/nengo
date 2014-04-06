import numpy as np

import logging

import nengo
from nengo.utils.distributions import Uniform, UniformHypersphere

logger = logging.getLogger(__name__)


class AutoAssociative(nengo.Network):
    """Store (key, value) associations, and lookup the value by key.

    This network implements a data structure that resembles a Python dict.
    (key, value) pairs can be stored, and values later retrieved by key.

    Voja's rule is used to adapt to the keys. And PES is used to associate
    those keys with their corresponding values.

    The main assumptions are that the keys are unit length, and dissimilar.

    In general, to store max_capacity keys in this dictionary, you must
    minimize the largest np.dot(k1, k2) across all max_capacity keys. This is
    not necessarily related to orthogonality of the keys.

    If the keys are expected to be closer together than optimal, then
    max_capactiy should be raised or else the first keys to be presented will
    steal the encoders that would have been used for the keys that are close
    together. When this occurs, everything will still work, but the later
    values will be represented less accurately. In this case, you can use the
    helper function Dict.calculate_max_capacity(d_key, c) to determine what
    the max_capacity should be given that you want the intercept to be c
    (i.e. given that you want the "influence" of each key to include all
    encoders e such that np.dot(key, e) >= c). Theoretically, a good value to
    use here would be np.cos(np.arccos(worst_dot_product) / 2), assuming a
    radius of 1, where worst_dot_product is the largest np.dot(k1, k2) across
    all pairs of keys. Using this intercept, which is half the angle between
    the worst pair, will ensure the best case that the influence of all keys
    are completely disjoint. This is useful for two additional reasons:

        - This tells you the theoretical max capacity assuming your keys
          continue to be tiled subject to this worst_dot_product.

        - This allows you to fix the representational accuracy, by setting the
          number of neurons in the memory layer to
          (max_capacity * neurons_per_item), which will make approximately
          neurons_per_item represent each association.

    Note that there is no perfect allocation of neurons to keys, since one
    cannot partition the surface of a sphere into circles without there being
    some overlap or missing surface.

    Refer to examples/associative_memory.ipynb for an example of how to use
    this Network, and a visually guided examination of the above subtleties.

    Parameters
    ----------
    neurons : Neurons
        Neurons to use for the memory layer. Number affects recall of values.
    max_capacity : int
        The tuning curves of the memory layer will be optimized to make
        (neurons.n_neurons / max_capacity) neurons respond to a random x.
        So this becomes the theoretical max capacity assuming the keys do a
        good job at tiling the space (discussion above).
    d_key : int, optional
        Dimensionality of the keys. Defaults to dimension of initial_keys if
        given.
    d_value : int, optional
        Dimensionality of the values. Defaults to dimension of initial_values
        if given.
    initial_keys : array_like, shape (num_keys, d_key), optional
        Initial keys used to bootstrap the encoders. Use this if some of the
        keys are known ahead of time. This is equivalent to presenting each key
        in sequence for an infinite period of time, in advance. Space that is
        not influenced by these keys will still be distributed uniformly, as
        usual. Defaults to None (uniformly distributes all encoders).
    initial_values : array_like, shape (len(initial_keys), d_value), optional
        Initial values that correspond to each initial key. If given, its
        length must equal len(initial_keys). These are used to optimize the
        initial decoders. Defaults to None (uses the default_value).
    default_value : array_like, shape (d_value), optional
        Default initial value to be used when initial_values is None, and for
        parts of the space that are not affected by the initial_keys.
        Defaults to the zero vector.
    n_error : int, optional
        Number of neurons to use in the error ensemble. Defaults to 200.
    voja_learning_rate : float, optional
        Learning rate for Voja's rule. Defaults to 1e-2.
    voja_filter : float, optional
        Post-synaptic filter on the memory layer's activity as input to
        Voja's rule. Defaults to 0.001. Determines how quickly the network
        will adapt to new keys.
    pes_learning_rate : float, optional
        Learning rate for the PES rule. Defaults to 1e-4. Determines how
        quickly the netork will associate a value with the given key.
    intercept_spread : float, optional
        The radius of uniform randomness to add to the intercept used for all
        of the encoders. Defaults to 0.05.
    n_dopamine : int, optional
        Number of neurons to use in the dopamine ensembles. Defaluts to 50.
        Lower values prevent the learning signal from being consistently
        transmitted when dopamine_filter is small.
    dopamine_strength : float, optional
        Strength of inhibitory signals in dopamine ensembles. Defaults to 20.
    dopamine_filter : float, optional
        Post-synaptic filter for connections from dopamine ensembles. Defaults
        to 0.001. Lower values make changes to the learning input propagate
        more instantly to the learning rules.
    dopamine_intercepts : Distribution, optional
        Distribution of intercepts for dopamine ensembles. Defaults to
        Uniform(0.1, 1), which ensures that the neurons do not fire when
        learning = 0.
    always_learn : bool, optional
        Specifies whether the memory should always be learning from the given
        keys and values. Defaults to False. Connect to the learning Node to
        make this dynamic (refer to the learning attribute documentation).
    voja_disable : bool, optional
        Set to True to disable Voja's rule. This will not break the dict, but
        it will impact recall accuracy, since some keys may share encoders.
        This is intended for debugging/validation purposes.

    Attributes
    ----------
    key : Node
        Input node for the current key. Can be 0.
    value : Node
        Input node for the current value. Can be anything if key is 0 or
        learning is off. Otherwise, it is the value that will become associated
        with the current key.
    output : Node
        Output node for the value which corresponds to the current key in
        memory. If the key doesn't exist, then the value will be arbitrary.
        Note that when connecting from output, you should use filter=None,
        otherwise you will get an additional layer of filtering.
    has_key : Node
        Outputs whether the dictionary has been shown this key while
        learning was turned on.
        Note that when connecting from has_key, you should use filter=None,
        otherwise you will get an additional layer of filtering.
    learning : Node
        If always_learn is False (the default), then this is an input node
        which scales the learning_rate (for Voja) and uninhibits the error
        signal (for PES). Set this signal to 0 when you want learning to be
        turned off (so that lookups can be done regardless of the current
        value), and to 1 when you want learning to be on (to store
        associations). Note: Accuracy may be slightly better if you can shut
        off learning a couple ms before lookup begins (not simultaneously),
        especially if the pes_learning_rate is high. A small dopamine_filter
        helps make these changes more instantaneous.
    intercept : float
        The intercept used for all of the encoders in order to make
        (neurons.n_neurons / max_capacity) of them respond to any randomly
        chosen key.
    memory : Ensemble
        Layer which stores all of the associations. Its encoders remember the
        given keys, and its decoders produce their corresponding values.
    dopamine : Ensemble
        Ensemble representing the learning node's value. Used by Voja's rule to
        scale the learning_rate.
    nopamine : Ensemble
        "No dopamine"; 0 when dopamine is sufficiently high, otherwise 1.
    value_error : Ensemble
        Represents the current error between the given value and the stored
        value. If nopamine is 1 (learning is off), then the error is inhibited
        to 0, so that PES will not learn the given value.
    has_key_error : Ensemble
        Represents the current error between the desired has_key and the
        stored has_key. Gated by same mechanism as value_error.
    voja : Voja
        Instance of the Voja learning rule.
    value_pes : PES
        Instance of the PES learning rule used to minimize value_error
        (learn output).
    has_key_pes : PES
        Instance of the PES learning rule used to minimize has_key_error
        (learn has_key).
    """

    def __init__(self, neurons, max_capacity, d_key=None, d_value=None,
                 initial_keys=None, initial_values=None, default_value=None,
                 n_error=200, voja_learning_rate=1e-2, voja_filter=0.001,
                 pes_learning_rate=1e-4, intercept_spread=0.05,
                 n_dopamine=50, dopamine_strength=20, dopamine_filter=0.001,
                 dopamine_intercepts=Uniform(0.1, 1), always_learn=False,
                 voja_disable=False):
        if max_capacity <= 0:
            raise ValueError("max_capacity (%d) must be positive" %
                             max_capacity)

        # Normalize initial_keys
        if initial_keys is None:
            initial_keys = []
        #initial_keys = np.asarray(
        #    [e / np.linalg.norm(e) for e in initial_keys])
        if len(initial_keys) > max_capacity:
            logger.warning("Length of initial_keys (%d) exceeds given "
                           "max_capacity (%s)",
                           len(initial_keys), max_capacity)

        # Infer d_key and d_value
        if d_key is None:
            if not len(initial_keys):
                raise ValueError("d_key must be specified if initial_keys "
                                   "is not specified")
            d_key = len(initial_keys[0])
        if d_value is None:
            if not len(initial_values):
                raise ValueError("d_value must be specified if "
                                   "initial_values is not specified")
            d_value = len(initial_values[0])

        # Set initial values to the default value if not given
        if default_value is None:
            default_value = np.zeros(d_value)
        if initial_values is None:
            initial_values = np.repeat(
                [default_value], len(initial_keys), axis=0)
        elif len(initial_values) != len(initial_keys):
            raise ValueError("initial_values must be None or same length "
                             "(%d) as initial_keys (%d)" %
                             (len(initial_values), len(initial_keys)))

        # Create input and output passthrough nodes
        self.key = nengo.Node(size_in=d_key, label="key")
        self.value = nengo.Node(size_in=d_value, label="value")
        self.output = nengo.Node(size_in=d_value, label="output")
        self.has_key = nengo.Node(size_in=1, label="has_key")
        self.learning = nengo.Node(
            size_in=1, output=lambda t, x: (x, 1)[always_learn],
            label="learning")

        # Create node/ensembles for scaling the learning rate
        # TODO: Separate inputs for Voja learning and PES learning.
        self.dopamine = nengo.Ensemble(
            nengo.LIF(n_dopamine), 1, intercepts=dopamine_intercepts,
            encoders=[[1]]*n_dopamine, label="dopamine")
        self.nopamine = nengo.Ensemble(
            nengo.LIF(n_dopamine), 1, intercepts=dopamine_intercepts,
            encoders=[[1]]*n_dopamine, label="nopamine")

        # Create ensemble which acts as the dictionary. The encoders will
        # shift towards the keys with Voja's rule, and the decoders will
        # shift towards the values with the PES learning rule. Aim to have
        # (neurons.n_neurons / max_capacity) neurons fire for a random x.
        self.intercept = self._calculate_intercept(
            d_key, 1 / float(max_capacity))
        encoders, value_function, has_key_function = self._initial_memory(
            neurons.n_neurons, d_key,
            initial_keys, initial_values, default_value, self.intercept)
        intercepts = Uniform(
            self.intercept - intercept_spread,
            min(self.intercept + intercept_spread, 1.0))
        self.memory = nengo.Ensemble(
            neurons, d_key, encoders=encoders, intercepts=intercepts,
            label="memory")

        # Create the ensembles for calculating error * learning
        self.value_error = nengo.Ensemble(
            nengo.LIF(n_error), d_value, label="value_error")
        self.has_key_error = nengo.Ensemble(
            nengo.LIF(n_error), 1, label="has_key_error")

        # Connect the memory Ensemble to the output Node with PES(value_error)
        # and the has_key node with PES(has_key_error)
        # Use the values "assigned" to each encoder as the evaluation points
        self.value_pes = nengo.PES(
            self.value_error, learning_rate=pes_learning_rate,
            label="learn_value")
        nengo.Connection(
            self.memory, self.output, eval_points=encoders,
            function=value_function, learning_rule=self.value_pes)
        self.has_key_pes = nengo.PES(
            self.has_key_error, learning_rate=pes_learning_rate,
            label="learn_has_key")
        nengo.Connection(
            self.memory, self.has_key, eval_points=encoders,
            function=has_key_function, learning_rule=self.has_key_pes)

        # Connect the learning signal to the error populations
        nengo.Connection(self.learning, self.dopamine, filter=None)
        nengo.Connection(nengo.Node(output=[1], label="bias"), self.nopamine)
        self._inhibit(self.dopamine, self.nopamine,
                      amount=dopamine_strength, filter=dopamine_filter)
        self._inhibit(self.nopamine, self.value_error,
                      amount=dopamine_strength, filter=dopamine_filter)
        self._inhibit(self.nopamine, self.has_key_error,
                      amount=dopamine_strength, filter=dopamine_filter)

        # Connect the key Node to the memory Ensemble with voja's rule
        self.voja = (nengo.Voja(learning_rate=voja_learning_rate,
                                learning=self.dopamine, filter=voja_filter,
                                learning_filter=dopamine_filter,
                                label="learn_key")
                     if not voja_disable else None)
        nengo.Connection(
            self.key, self.memory, filter=None, learning_rule=self.voja)

        # Compute the value_error and has_key_error
        nengo.Connection(self.value, self.value_error, filter=None)
        nengo.Connection(
            self.output, self.value_error, filter=None, transform=-1)
        nengo.Connection(nengo.Node(output=[1]), self.has_key_error)
        nengo.Connection(
            self.has_key, self.has_key_error, filter=None, transform=-1)

    @classmethod
    def _initial_memory(cls, n, d_key, initial_keys, initial_values,
                        default_value, intercept):
        """Returns the initial encoders and functions to decode."""
        # Randomly generate encoders and then set them to the initial_keys, by
        # finding the first key which stimulates an encoder, for each encoder.
        encoders = UniformHypersphere(d_key, surface=True).sample(n)
        encoder_to_value = dict()
        encoder_to_has_key = dict()
        for e in encoders:
            for key, value in reversed(zip(initial_keys, initial_values)):
                if np.dot(e, key) >= intercept:
                    e[...] = key
                    encoder_to_value[tuple(e)] = value  # since e is mutable
                    encoder_to_has_key[tuple(e)] = 1
                    break
            else:
                # Use the default value for this encoder if it's not close to
                # any of the initial keys
                encoder_to_value[tuple(e)] = default_value
                encoder_to_has_key[tuple(e)] = 0
        return (encoders,
                lambda e: encoder_to_value[tuple(e)],
                lambda e: encoder_to_has_key[tuple(e)])

    @classmethod
    def _calculate_intercept(cls, d, p):
        """Returns c such that np.dot(u, v) >= c with probability p.

        Here, u and v are two randomly generated vectors of dimension d.
        This works by the following formula, (1 - x**2)**((d - 3)/2.0), which
        gives the probability that a coordinate of a random point on a
        hypersphere is equal to x.

        The probability distribution of the dot product of two randomly chosen
        vectors is equivalent to the above, since we can always rotate the
        sphere such that one of the vectors is a unit vector, and then the
        dot product just becomes the component corresponding to that unit
        vector.

        This can be used to find the intercept such that a randomly generated
        encoder will fire in response to a random input x with probability p.
        """
        x, cpx = cls._component_probability_dist(d)
        return x[cpx >= 1 - p][0]

    @classmethod
    def calculate_max_capacity(cls, d, c):
        """Calculates what max_capacity should be to achieve desired intercept.

        This is the inverse of _calculate_intercept. Given some number of
        dimensions d, returns the max_capacity such that
        _calculate_intercepts(d, 1.0 / max_capacity) == c.
        """
        x, cpx = cls._component_probability_dist(d)
        return 1.0 / (1 - cpx[x >= c][0])

    @classmethod
    def _component_probability_dist(cls, d, dx=0.001):
        """Returns x and py such that probability of component <= x is cpx."""
        x = np.arange(-1+dx, 1, dx)
        cpx = ((1 - x**2)**((d - 3)/2.0)).cumsum()
        cpx = cpx / sum(cpx) / dx
        return x, cpx

    @classmethod
    def _inhibit(cls, pre, post, amount, **kwargs):
        """Creates a connection which inhibits post whenever pre fires."""
        return nengo.Connection(
            pre.neurons, post.neurons,
            transform=-amount*np.ones((post.n_neurons, pre.n_neurons)),
            **kwargs)


class HeteroAssociative(AutoAssociative):
    """Store keys and recall them later with noisy keys.

    Behaves identically to the AutoAssociative memory, where the values are
    the keys.
    """

    def __init__(self, neurons, max_capacity, d_key=None, initial_keys=None,
                 **kwargs):
        super(HeteroAssociative, self).make(
            neurons, max_capacity, d_key, d_key, initial_keys, initial_keys,
            **kwargs)
        nengo.Connection(self.key, self.value, filter=None)
