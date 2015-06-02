import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.operator import (
    DotInc, ElementwiseInc, Operator, Reset, PreserveValue)
from nengo.builder.signal import Signal
from nengo.connection import LearningRule
from nengo.ensemble import Ensemble, Neurons
from nengo.learning_rules import BCM, Oja, PES, Voja
from nengo.node import Node
from nengo.synapses import Lowpass


class SimBCM(Operator):
    """Calculate delta omega according to the BCM rule."""
    def __init__(self, pre_filtered, post_filtered, theta, delta,
                 learning_rate, tag=None):
        self.post_filtered = post_filtered
        self.pre_filtered = pre_filtered
        self.theta = theta
        self.delta = delta
        self.learning_rate = learning_rate
        self.tag = tag

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, theta]
        self.updates = [delta]

    def __str__(self):
        return 'SimBCM(pre=%s, post=%s -> %s%s)' % (
            self.pre_filtered, self.post_filtered, self.delta, self._tagstr)

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        theta = signals[self.theta]
        delta = signals[self.delta]
        alpha = self.learning_rate * dt

        def step_simbcm():
            delta[...] = np.outer(
                alpha * post_filtered * (post_filtered - theta), pre_filtered)
        return step_simbcm


class SimOja(Operator):
    """Calculate delta omega according to the Oja rule."""
    def __init__(self, pre_filtered, post_filtered, weights, delta,
                 learning_rate, beta, tag=None):
        self.post_filtered = post_filtered
        self.pre_filtered = pre_filtered
        self.weights = weights
        self.delta = delta
        self.learning_rate = learning_rate
        self.beta = beta
        self.tag = tag

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, weights]
        self.updates = [delta]

    def __str__(self):
        return 'SimOja(pre=%s, post=%s -> %s%s)' % (
            self.pre_filtered, self.post_filtered, self.delta, self._tagstr)

    def make_step(self, signals, dt, rng):
        weights = signals[self.weights]
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        delta = signals[self.delta]
        alpha = self.learning_rate * dt
        beta = self.beta

        def step_simoja():
            # perform forgetting
            post_squared = alpha * post_filtered * post_filtered
            delta[...] = -beta * weights * post_squared[:, None]

            # perform update
            delta[...] += np.outer(alpha * post_filtered, pre_filtered)

        return step_simoja


def get_pre_ens(conn):
    return (conn.pre_obj if isinstance(conn.pre_obj, Ensemble)
            else conn.pre_obj.ensemble)


def get_post_ens(conn):
    return (conn.post_obj if isinstance(conn.post_obj, (Ensemble, Node))
            else conn.post_obj.ensemble)


@Builder.register(LearningRule)
def build_learning_rule(model, rule):
    conn = rule.connection
    rule_type = rule.learning_rule_type

    # --- Set up delta signal and += transform / decoders
    if rule_type.modifies == 'encoders':
        post = get_post_ens(conn)
        delta = Signal(
            np.zeros((post.n_neurons, post.dimensions)), name='Delta')
        modifies_sig = model.sig[post]['encoders']
        model.add_op(PreserveValue(modifies_sig))
    else:
        pre = get_pre_ens(conn)
        if not conn.is_factored:
            post = get_post_ens(conn)
            delta = Signal(
                np.zeros((post.n_neurons, pre.n_neurons)), name='Delta')
        else:
            delta = Signal(
                np.zeros((rule.size_in, pre.n_neurons)), name='Delta')
        modifies_sig = model.sig[conn]['weights']

    model.add_op(ElementwiseInc(
        model.sig['common'][1], delta, modifies_sig, tag="omega += delta"))
    model.sig[rule]['delta'] = delta
    model.build(rule_type, rule)  # Updates delta


@Builder.register(BCM)
def build_bcm(model, bcm, rule):
    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]['out']
    pre_filtered = model.build(Lowpass(bcm.pre_tau), pre_activities)
    post_activities = model.sig[get_post_ens(conn).neurons]['out']
    post_filtered = model.build(Lowpass(bcm.post_tau), post_activities)
    theta = model.build(Lowpass(bcm.theta_tau), post_filtered)

    model.add_op(SimBCM(pre_filtered,
                        post_filtered,
                        theta,
                        model.sig[rule]['delta'],
                        learning_rate=bcm.learning_rate))

    # expose these for probes
    model.sig[rule]['theta'] = theta
    model.sig[rule]['pre_filtered'] = pre_filtered
    model.sig[rule]['post_filtered'] = post_filtered

    model.params[rule] = None  # no build-time info to return


@Builder.register(Oja)
def build_oja(model, oja, rule):
    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]['out']
    post_activities = model.sig[get_post_ens(conn).neurons]['out']
    pre_filtered = model.build(Lowpass(oja.pre_tau), pre_activities)
    post_filtered = model.build(Lowpass(oja.post_tau), post_activities)

    model.add_op(SimOja(pre_filtered,
                        post_filtered,
                        model.sig[conn]['weights'],
                        model.sig[rule]['delta'],
                        learning_rate=oja.learning_rate,
                        beta=oja.beta))

    # expose these for probes
    model.sig[rule]['pre_filtered'] = pre_filtered
    model.sig[rule]['post_filtered'] = post_filtered

    model.params[rule] = None  # no build-time info to return


@Builder.register(Voja)
def build_voja(model, voja, rule):
    conn = rule.connection

    # Input vector, decoded and filtered from pre
    x = model.sig[conn]['out']

    # Filtered post activity
    post = conn.post_obj
    if voja.post_tau is not None:
        post_filtered = model.build(
            Lowpass(voja.pos_tau), model.sig[post]['out'])
    else:
        post_filtered = model.sig[post]['out']

    # Learning signal, defaults to 1 in case no connection is made
    # and multiplied by the learning_rate * dt
    learning = Signal(np.zeros(rule.size_in), name="Voja:learning")
    assert rule.size_in == 1
    model.add_op(Reset(learning, value=1.0))
    model.sig[rule]['in'] = learning  # optional connection will attach here
    learning_rate = voja.learning_rate * model.dt
    lr_sig = Signal(learning_rate, name="Voja:learning_rate")

    # Post encoders (scaled)
    encoders = model.sig[post]['encoders']
    # The gain and radius are folded into the encoders during the ensemble
    # build process, so we need to make sure that the deltas are proportional
    # to this scaling factor
    encoder_scale = Signal(
        model.params[post].gain / post.radius, name="Voja:encoder_scale")
    assert post_filtered.shape == encoder_scale.shape

    # Intermediary signals
    learning_scaled = Signal(np.zeros(1), name="Voja:learning_scaled")
    model.add_op(Reset(learning_scaled))

    outer = Signal(np.zeros(encoders.shape), name="Voja:outer")
    model.add_op(Reset(outer))

    normalize = Signal(np.zeros(post_filtered.shape), name="Voja:normalize")
    model.add_op(Reset(normalize))

    delta = Signal(np.zeros(encoders.shape), name="Voja:delta")
    model.add_op(Reset(delta))

    # Operations to perform the voja update rule
    # This implements `learning * (outer(y, x) - y*e)`, where
    #    `learning` is the `learning_rate * learning signal` (modulatory input)
    #    `y` is the filtered post-synaptic activity (tau = `post_tau`)
    #    `x` is the decoded vector
    #    `e` are the encoders
    # This is the same as Oja's rule, `learning * (outer(y, a) - s*y^2*W)`, if
    #    `e` <-> `W`    (encoders <-> weights)
    #    `a` <-> `x`    (activities <-> vector)
    #    `s` <-> `1/y`  (dynamic normalization)

    # Computes outer(y, x), and adds to delta signal
    # encoder_scale introduced here to account for scaling of the encoders
    model.add_op(ElementwiseInc(
        post_filtered.column(), x.row(), outer, tag="Voja:outer"))
    model.add_op(ElementwiseInc(
        encoder_scale.column(), outer, delta, tag="Voja:add_outer"))

    # Computes -y^2 e, and adds to delta signal
    model.add_op(
        ElementwiseInc(Signal(-1), post_filtered, normalize,
                       tag="Voja:normalize"))
    model.add_op(
        ElementwiseInc(normalize.column(), encoders, delta,
                       tag="Voja:add_normalize"))

    # Computes learning, and uses it to scale the delta update to the encoders
    model.add_op(
        ElementwiseInc(learning, lr_sig, learning_scaled,
                       tag="Voja:learning_scaled"))

    model.add_op(Reset(model.sig[rule]['delta']))
    model.add_op(
        ElementwiseInc(delta, learning_scaled, model.sig[rule]['delta'],
                       tag="Voja:inc_update"))

    model.sig[rule]['scaled_encoders'] = encoders
    model.sig[rule]['post_filtered'] = post_filtered

    model.params[rule] = None  # no build-time info to return


@Builder.register(PES)
def build_pes(model, pes, rule):
    conn = rule.connection

    # Create input error signal
    error = Signal(np.zeros(rule.size_in), name="PES:error")
    model.add_op(Reset(error))
    model.sig[rule]['in'] = error  # error connection will attach here

    acts = model.build(Lowpass(pes.pre_tau), model.sig[conn.pre_obj]['out'])
    acts_view = acts.row()

    # Compute the correction, i.e. the scaled negative error
    correction = Signal(np.zeros(error.shape), name="PES:correction")
    local_error = correction.column()
    model.add_op(Reset(correction))

    # correction = -learning_rate * (dt / n_neurons) * error
    n_neurons = (conn.pre_obj.n_neurons if isinstance(conn.pre_obj, Ensemble)
                 else conn.pre_obj.size_in)
    lr_sig = Signal(-pes.learning_rate * model.dt / n_neurons,
                    name="PES:learning_rate")
    model.add_op(DotInc(lr_sig, error, correction, tag="PES:correct"))

    if not conn.is_factored:
        post = get_post_ens(conn)
        weights = model.sig[conn]['weights']
        encoders = model.sig[post]['encoders']

        # encoded = dot(encoders, correction)
        encoded = Signal(np.zeros(weights.shape[0]), name="PES:encoded")
        model.add_op(Reset(encoded))
        model.add_op(DotInc(encoders, correction, encoded, tag="PES:encode"))
        local_error = encoded.column()
    elif not isinstance(conn.pre_obj, (Ensemble, Neurons)):
        raise ValueError("'pre' object '%s' not suitable for PES learning"
                         % (conn.pre_obj))

    # delta = local_error * activities
    model.add_op(Reset(model.sig[rule]['delta']))
    model.add_op(ElementwiseInc(
        local_error, acts_view, model.sig[rule]['delta'], tag="PES:Inc Delta"))

    # expose these for probes
    model.sig[rule]['error'] = error
    model.sig[rule]['correction'] = correction
    model.sig[rule]['activities'] = acts

    model.params[rule] = None  # no build-time info to return
