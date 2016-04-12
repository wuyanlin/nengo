import collections
import logging

import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.ensemble import gen_eval_points, get_activities
from nengo.builder.node import SimPyFunc
from nengo.builder.operator import (
    DotInc, ElementwiseInc, PreserveValue, Reset, SlicedCopy)
from nengo.builder.signal import Signal
from nengo.connection import Connection
from nengo.dists import Distribution
from nengo.ensemble import Ensemble, Neurons
from nengo.exceptions import BuildError, ObsoleteError
from nengo.neurons import Direct
from nengo.node import Node
from nengo.utils.compat import is_iterable, itervalues

logger = logging.getLogger(__name__)

BuiltConnection_ = collections.namedtuple(
    'BuiltConnection', ['eval_points', 'solver_info', 'weights', 'transform'])


class BuiltConnection(BuiltConnection_):
    """Subclassing the namedtuple to provide better error messages."""
    @property
    def decoders(self):
        raise ObsoleteError("decoders are now part of 'weights'. "
                            "Access BuiltConnection.weights instead.",
                            since="v2.1.0")


def get_eval_points(model, conn, rng):
    if conn.eval_points is None:
        view = model.params[conn.pre_obj].eval_points.view()
        view.setflags(write=False)
        return view
    else:
        return gen_eval_points(
            conn.pre_obj, conn.eval_points, rng, conn.scale_eval_points)


def get_targets(model, conn, eval_points):
    if conn.function is None:
        targets = eval_points[:, conn.pre_slice]
    elif isinstance(conn.function, np.ndarray):
        targets = conn.function
    else:
        targets = np.zeros((len(eval_points), conn.size_mid))
        for i, ep in enumerate(eval_points[:, conn.pre_slice]):
            targets[i] = conn.function(ep)

    return targets


def build_linear_system(model, conn, rng):
    eval_points = get_eval_points(model, conn, rng)
    activities = get_activities(model, conn.pre_obj, eval_points)
    if np.count_nonzero(activities) == 0:
        raise BuildError(
            "Building %s: 'activites' matrix is all zero for %s. "
            "This is because no evaluation points fall in the firing "
            "ranges of any neurons." % (conn, conn.pre_obj))

    targets = get_targets(model, conn, eval_points)
    return eval_points, activities, targets


def build_decoders(model, conn, rng, transform):
    encoders = model.params[conn.pre_obj].encoders
    gain = model.params[conn.pre_obj].gain
    bias = model.params[conn.pre_obj].bias

    eval_points = get_eval_points(model, conn, rng)
    targets = get_targets(model, conn, eval_points)

    x = np.dot(eval_points, encoders.T / conn.pre_obj.radius)
    E = None
    if conn.solver.weights:
        E = model.params[conn.post_obj].scaled_encoders.T[conn.post_slice]
        # include transform in solved weights
        targets = multiply(targets, transform.T)

    try:
        wrapped_solver = (model.decoder_cache.wrap_solver(solve_for_decoders)
                          if model.seeded[conn] else solve_for_decoders)
        decoders, solver_info = wrapped_solver(
            conn.solver, conn.pre_obj.neuron_type, gain, bias, x, targets,
            rng=rng, E=E)
    except BuildError:
        raise BuildError(
            "Building %s: 'activities' matrix is all zero for %s. "
            "This is because no evaluation points fall in the firing "
            "ranges of any neurons." % (conn, conn.pre_obj))

    weights = (decoders.T if conn.solver.weights else
               multiply(transform, decoders.T))
    return eval_points, weights, solver_info


def solve_for_decoders(
        solver, neuron_type, gain, bias, x, targets, rng, E=None):
    # compute activities in blocks to save memory
    block_size = 100000
    block_len = max(int(float(block_size) / np.prod(x.shape[1:])), 1)
    activities = np.zeros_like(x)
    for i in range(int(np.ceil(float(x.shape[0]) / block_len))):
        i0, i1 = i*block_len, min((i+1)*block_len, x.shape[0])
        neuron_type.rates(x[i0:i1], gain, bias, out=activities[i0:i1])

    if np.count_nonzero(activities) == 0:
        raise BuildError()

    logger.debug("Solving system: %s" % (activities.shape,))
    if solver.weights:
        decoders, solver_info = solver(activities, targets, rng=rng, E=E)
    else:
        decoders, solver_info = solver(activities, targets, rng=rng)

    return decoders, solver_info


def multiply(x, y):
    if x.ndim <= 2 and y.ndim < 2:
        return x * y
    elif x.ndim < 2 and y.ndim == 2:
        return x.reshape(-1, 1) * y
    elif x.ndim == 2 and y.ndim == 2:
        return np.dot(x, y)
    else:
        raise BuildError("Tensors not supported (x.ndim = %d, y.ndim = %d)"
                         % (x.ndim, y.ndim))


def slice_signal(model, signal, sl):
    assert signal.ndim == 1
    if isinstance(sl, slice) and (sl.step is None or sl.step == 1):
        return signal[sl]
    else:
        size = np.arange(signal.size)[sl].size
        sliced_signal = Signal(np.zeros(size), name="%s.sliced" % signal.name)
        model.add_op(SlicedCopy(signal, sliced_signal, a_slice=sl))
        return sliced_signal


@Builder.register(Connection)  # noqa: C901
def build_connection(model, conn):
    # Create random number generator
    rng = np.random.RandomState(model.seeds[conn])

    # Get input and output connections from pre and post
    def get_prepost_signal(is_pre):
        target = conn.pre_obj if is_pre else conn.post_obj
        key = 'out' if is_pre else 'in'

        if target not in model.sig:
            raise BuildError("Building %s: the %r object %s is not in the "
                             "model, or has a size of zero."
                             % (conn, 'pre' if is_pre else 'post', target))
        if key not in model.sig[target]:
            raise BuildError(
                "Building %s: the %r object %s has a %r size of zero."
                % (conn, 'pre' if is_pre else 'post', target, key))

        return model.sig[target][key]

    model.sig[conn]['in'] = get_prepost_signal(is_pre=True)
    model.sig[conn]['out'] = get_prepost_signal(is_pre=False)

    weights = None
    eval_points = None
    solver_info = None
    signal_size = conn.size_out
    post_slice = conn.post_slice

    # Sample transform if given a distribution
    transform = (conn.transform.sample(conn.size_out, conn.size_mid, rng=rng)
                 if isinstance(conn.transform, Distribution) else
                 np.array(conn.transform))

    # Figure out the signal going across this connection
    in_signal = model.sig[conn]['in']
    if (isinstance(conn.pre_obj, Node) or
            (isinstance(conn.pre_obj, Ensemble) and
             isinstance(conn.pre_obj.neuron_type, Direct))):
        # Node or Decoded connection in directmode
        weights = transform
        sliced_in = slice_signal(model, in_signal, conn.pre_slice)
        if conn.function is None:
            in_signal = sliced_in
        elif isinstance(conn.function, np.ndarray):
            raise BuildError("Cannot use function points in direct connection")
        else:
            in_signal = Signal(np.zeros(conn.size_mid), name='%s.func' % conn)
            model.add_op(SimPyFunc(in_signal, conn.function, None, sliced_in))
    elif isinstance(conn.pre_obj, Ensemble):  # Normal decoded connection
        eval_points, weights, solver_info = build_decoders(
            model, conn, rng, transform)
        if conn.solver.weights:
            model.sig[conn]['out'] = model.sig[conn.post_obj.neurons]['in']
            signal_size = conn.post_obj.neurons.size_in
            post_slice = Ellipsis  # don't apply slice later
    else:
        weights = transform
        in_signal = slice_signal(model, in_signal, conn.pre_slice)

    if isinstance(conn.post_obj, Neurons):
        weights = multiply(
            model.params[conn.post_obj.ensemble].gain[post_slice], weights)

    # Add operator for applying weights
    model.sig[conn]['weights'] = Signal(
        weights, name="%s.weights" % conn, readonly=True)
    signal = Signal(np.zeros(signal_size), name="%s.weighted" % conn)
    model.add_op(Reset(signal))
    op = ElementwiseInc if weights.ndim < 2 else DotInc
    model.add_op(op(model.sig[conn]['weights'],
                    in_signal,
                    signal,
                    tag="%s.weights_elementwiseinc" % conn))

    # Add operator for filtering
    if conn.synapse is not None:
        signal = model.build(conn.synapse, signal)

    # Store the weighted-filtered output in case we want to probe it
    model.sig[conn]['weighted'] = signal

    # Copy to the proper slice
    model.add_op(SlicedCopy(
        signal, model.sig[conn]['out'], b_slice=post_slice,
        inc=True, tag="%s.gain" % conn))

    # Build learning rules
    if conn.learning_rule is not None:
        rule = conn.learning_rule
        rule = [rule] if not is_iterable(rule) else rule
        targets = []
        for r in itervalues(rule) if isinstance(rule, dict) else rule:
            model.build(r)
            targets.append(r.modifies)

        if 'encoders' in targets:
            encoder_sig = model.sig[conn.post_obj]['encoders']
            if not any(isinstance(op, PreserveValue) and op.dst is encoder_sig
                       for op in model.operators):
                encoder_sig.readonly = False
                model.add_op(PreserveValue(encoder_sig))
        if 'decoders' in targets or 'weights' in targets:
            if weights.ndim < 2:
                raise BuildError(
                    "'transform' must be a 2-dimensional array for learning")
            model.sig[conn]['weights'].readonly = False
            model.add_op(PreserveValue(model.sig[conn]['weights']))

    model.params[conn] = BuiltConnection(eval_points=eval_points,
                                         solver_info=solver_info,
                                         transform=transform,
                                         weights=weights)
