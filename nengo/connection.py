import logging
import weakref

import numpy as np

from nengo.base import NengoObject, NengoObjectParam, ObjView
from nengo.dists import Distribution, DistOrArrayParam
from nengo.ensemble import Ensemble, Neurons
from nengo.exceptions import ValidationError
from nengo.learning_rules import LearningRuleType, LearningRuleTypeParam
from nengo.node import Node
from nengo.params import (Default, Unconfigurable, ObsoleteParam,
                          BoolParam, FunctionParam)
from nengo.solvers import LstsqL2, SolverParam
from nengo.synapses import Lowpass, SynapseParam
from nengo.utils.compat import is_array_like, is_iterable, iteritems

logger = logging.getLogger(__name__)


class PrePostParam(NengoObjectParam):
    def validate(self, conn, nengo_obj):
        super(PrePostParam, self).validate(conn, nengo_obj)
        if isinstance(nengo_obj, Connection):
            raise ValidationError(
                "Cannot connect to or from connections. "
                "Did you mean to connect to the connection's learning rule?",
                attr=self.name, obj=conn)


class ConnectionLearningRuleTypeParam(LearningRuleTypeParam):
    """Connection-specific validation for learning rules."""

    def __set__(self, conn, rule):
        conn._learning_rule = None
        super(ConnectionLearningRuleTypeParam, self).__set__(conn, rule)

    def validate_rule(self, conn, rule):
        super(ConnectionLearningRuleTypeParam, self).validate_rule(conn, rule)

        # --- Check pre object
        if rule.modifies in ('decoders', 'weights'):
            # pre object must be neural
            if not isinstance(conn.pre_obj, (Ensemble, Neurons)):
                raise ValidationError(
                    "pre' must be of type 'Ensemble' or 'Neurons' for "
                    "learning rule '%s' (got type %r)" % (
                        rule, type(conn.pre_obj).__name__),
                    attr=self.name, obj=conn)

        # --- Check post object
        if rule.modifies == 'encoders':
            if not isinstance(conn.post_obj, Ensemble):
                raise ValidationError(
                    "'post' must be of type 'Ensemble' (got %r) "
                    "for learning rule '%s'"
                    % (type(conn.pre_obj).__name__, rule),
                    attr=self.name, obj=conn)
        else:
            if not isinstance(conn.post_obj, (Ensemble, Neurons, Node)):
                raise ValidationError(
                    "'post' must be of type 'Ensemble', 'Neurons' or 'Node' "
                    "(got %r) for learning rule '%s'"
                    % (type(conn.post_obj).__name__, rule),
                    attr=self.name, obj=conn)

        if rule.modifies == 'weights':
            # If the rule modifies 'weights', then it must have full weights
            if conn.is_decoded:
                raise ValidationError(
                    "Learning rule '%s' can not be applied to decoded "
                    "connections. Try setting solver.weights to True or "
                    "connecting between two Neurons objects." % rule,
                    attr=self.name, obj=conn)

            # transform matrix must be 2D
            pre_size = (
                conn.pre_obj.n_neurons if isinstance(conn.pre_obj, Ensemble)
                else conn.pre.size_out)
            post_size = conn.post.size_in
            if (not conn.solver.weights and
                    conn.transform.shape != (post_size, pre_size)):
                raise ValidationError(
                    "Transform must be 2D array with shape post_neurons x "
                    "pre_neurons (%d, %d)" % (pre_size, post_size),
                    attr=self.name, obj=conn)


class ConnectionSolverParam(SolverParam):
    """Connection-specific validation for decoder solvers."""

    def validate(self, conn, solver):
        super(ConnectionSolverParam, self).validate(conn, solver)
        if solver is not None:
            if solver.weights and not isinstance(conn.pre_obj, Ensemble):
                raise ValidationError(
                    "weight solvers only work for connections from ensembles "
                    "(got %r)" % type(conn.pre_obj).__name__,
                    attr=self.name, obj=conn)
            if solver.weights and not isinstance(conn.post_obj, Ensemble):
                raise ValidationError(
                    "weight solvers only work for connections to ensembles "
                    "(got %r)" % type(conn.post_obj).__name__,
                    attr=self.name, obj=conn)


class EvalPointsParam(DistOrArrayParam):
    def validate(self, conn, distorarray):
        """Eval points are only valid when pre is an ensemble."""
        if not isinstance(conn.pre, Ensemble):
            msg = ("eval_points are only valid on connections from ensembles "
                   "(got type '%s')" % type(conn.pre).__name__)
            raise ValidationError(msg, attr=self.name, obj=conn)
        return super(EvalPointsParam, self).validate(conn, distorarray)


class ConnectionFunctionParam(FunctionParam):
    """Connection-specific validation for functions."""

    def __set__(self, conn, function):
        if function is None:
            function_info = self.Info(function=None, size=None)
        elif is_array_like(function):
            array = np.array(function, copy=False, dtype=np.float64)
            self.validate_array(conn, array)
            function_info = self.Info(function=array, size=array.shape[1])
        elif callable(function):
            function_info = self.Info(function=function,
                                      size=self.determine_size(conn, function))
            self.validate_callable(conn, function_info)
        else:
            raise ValidationError("Invalid connection function type %r "
                                  "(must be callable or array-like)"
                                  % type(function).__name__,
                                  attr=self.name, obj=conn)

        self.validate(conn, function_info)
        self.data[conn] = function_info

    def function_args(self, conn, function):
        x = (conn.eval_points[0] if is_iterable(conn.eval_points)
             else np.zeros(conn.size_in))
        return (x,)

    def validate_array(self, conn, ndarray):
        if not isinstance(conn.eval_points, np.ndarray):
            raise ValidationError(
                "Connection evaluation points must be explicitly set to be "
                "an array, to set function points",
                attr=self.name, obj=conn)

        if ndarray.ndim != 2:
            raise ValidationError("array must be 2D (got %dD)" % ndarray.ndim,
                                  attr=self.name, obj=conn)

        if ndarray.shape[0] != conn.eval_points.shape[0]:
            raise ValidationError(
                "Number of evaluation points must match number "
                "of function points (%d != %d)"
                % (ndarray.shape[0], conn.eval_points.shape[0]),
                attr=self.name, obj=conn)

    def validate_callable(self, conn, function_info):
        super(ConnectionFunctionParam, self).validate(conn, function_info)

    def validate(self, conn, function_info):
        function, size = function_info

        if function is not None:
            if not isinstance(conn.pre_obj, (Node, Ensemble)):
                raise ValidationError(
                    "function can only be set for connections from an Ensemble"
                    " or Node (got type %r)" % type(conn.pre_obj).__name__,
                    attr=self.name, obj=conn)

            if isinstance(conn.pre_obj, Node) and conn.pre_obj.output is None:
                raise ValidationError(
                    "Cannot apply functions to passthrough nodes",
                    attr=self.name, obj=conn)

        size_mid = conn.size_in if size is None else size
        transform = conn.transform
        type_pre = type(conn.pre_obj).__name__

        if isinstance(transform, np.ndarray):
            if transform.ndim < 2 and size_mid != conn.size_out:
                raise ValidationError(
                    "function output size is incorrect; should return a "
                    "vector of size %d" % conn.size_out, attr=self.name,
                    obj=conn)

            if transform.ndim == 2 and size_mid != transform.shape[1]:
                # check input dimensionality matches transform
                raise ValidationError(
                    "%s output size (%d) not equal to transform input size "
                    "(%d)" % (type_pre, size_mid, transform.shape[1]),
                    attr=self.name, obj=conn)


class TransformParam(DistOrArrayParam):
    """The transform additionally validates size_out."""

    def __init__(self, name, default, optional=False, readonly=False):
        super(TransformParam, self).__init__(
            name, default, (), optional, readonly)

    def validate(self, conn, transform):
        if not isinstance(transform, Distribution):
            # if transform is an array, figure out what the correct shape
            # should be
            transform = np.asarray(transform, dtype=np.float64)

            if transform.ndim == 0:
                self.shape = ()
            elif transform.ndim == 1:
                self.shape = ('size_out',)
            elif transform.ndim == 2:
                # Actually (size_out, size_mid) but Function handles size_mid
                self.shape = ('size_out', '*')

                # check for repeated dimensions in lists, as these don't work
                # for two-dimensional transforms
                def repeated_inds(x):
                    return (not isinstance(x, slice) and
                            np.unique(x).size != len(x))
                if repeated_inds(conn.pre_slice):
                    raise ValidationError(
                        "Input object selection has repeated indices",
                        attr=self.name, obj=conn)
                if repeated_inds(conn.post_slice):
                    raise ValidationError(
                        "Output object selection has repeated indices",
                        attr=self.name, obj=conn)
            else:
                raise ValidationError(
                    "Cannot handle transforms with dimensions > 2",
                    attr=self.name, obj=conn)

        super(TransformParam, self).validate(conn, transform)

        return transform


class Connection(NengoObject):
    """Connects two objects together.

    Almost any Nengo object can act as the pre or post side of a connection.
    Additionally, you can use Python slice syntax to access only some of the
    dimensions of the pre or post object.

    For example, if ``node`` has ``size_out=2`` and ``ensemble`` has
    ``size_in=1``, we could not create the following connection::

        nengo.Connection(node, ensemble)

    But, we could create either of these two connections.

        nengo.Connection(node[0], ensemble)
        nengo.Connection(ndoe[1], ensemble)

    Parameters
    ----------
    pre : Ensemble or Neurons or Node
        The source Nengo object for the connection.
    post : Ensemble or Neurons or Node or Probe
        The destination object for the connection.

    label : string
        A descriptive label for the connection.
    dimensions : int
        The number of output dimensions of the pre object, including
        `function`, but not including `transform`.
    eval_points : (n_eval_points, pre_size) array_like or int
        Points at which to evaluate `function` when computing decoders,
        spanning the interval (-pre.radius, pre.radius) in each dimension.
    synapse : float, optional
        Post-synaptic time constant (PSTC) to use for filtering.
    transform : (post_size, pre_size) array_like, optional
        Linear transform mapping the pre output to the post input.
        This transform is in terms of the sliced size; if either pre
        or post is a slice, the transform must be of shape
        (len(pre_slice), len(post_slice)).
    solver : Solver
        Instance of a Solver class to compute decoders or weights
        (see `nengo.solvers`). If `solver.weights` is True, a full
        connection weight matrix is computed instead of decoders.
    function : callable or array_like, optional
        Function to compute using the pre population (pre must be Ensemble).
        If an array is passed, the function is implicitly defined by the
        provided points and the provided eval_points.
    eval_points : (n_eval_points, pre_size) array_like or int, optional
        Points at which to evaluate `function` when computing decoders,
        spanning the interval (-pre.radius, pre.radius) in each dimension.
    scale_eval_points : bool
        Indicates whether the eval_points should be scaled by the radius of
        the pre Ensemble. Defaults to True.
    learning_rule_type : instance or list or dict of LearningRuleType, optional
        Methods of modifying the connection weights during simulation.

    Attributes
    ----------
    dimensions : int
        The number of output dimensions of the pre object, including
        `function`, but before applying the `transform`.
    function : callable
        The given function.
    function_size : int
        The output dimensionality of the given function. Defaults to 0.
    is_decoded: bool
        True if and only if the connection is decoded. This will not occur
        when `solver.weights` is True or both `pre` and `post` are `Neurons`.
    label : str
        A human-readable connection label for debugging and visualization.
        Incorporates the labels of the pre and post objects.
    learning_rule : LearningRule or collection of LearningRule
        The LearningRule objects corresponding to `learning_rule_type`, and in
        the same format. Use these to probe the learning rules.
    learning_rule_type : instance or list or dict of LearningRuleType, optional
        The learning rule types.
    post : Ensemble or Neurons or Node or Probe
        The given pre object.
    pre : Ensemble or Neurons or Node
        The given pre object.
    transform : (post_size, pre_size) array_like
        Linear transform mapping the pre output to the post input.
    seed : int
        The seed used for random number generation.
    """

    pre = PrePostParam('pre', nonzero_size_out=True)
    post = PrePostParam('post', nonzero_size_in=True)
    synapse = SynapseParam('synapse', default=Lowpass(0.005))
    transform = TransformParam('transform', default=np.array(1.0))
    solver = ConnectionSolverParam('solver', default=LstsqL2())
    learning_rule_type = ConnectionLearningRuleTypeParam(
        'learning_rule_type', default=None, optional=True)
    function_info = ConnectionFunctionParam(
        'function', default=None, optional=True)
    eval_points = EvalPointsParam('eval_points',
                                  default=None,
                                  optional=True,
                                  sample_shape=('*', 'size_in'))
    scale_eval_points = BoolParam('scale_eval_points', default=True)
    modulatory = ObsoleteParam(
        'modulatory',
        "Modulatory connections have been removed. "
        "Connect to a learning rule instead.",
        since="v2.1.0",
        url="https://github.com/nengo/nengo/issues/632#issuecomment-71663849")

    def __init__(self, pre, post, synapse=Default, transform=Default,
                 solver=Default, learning_rule_type=Default, function=Default,
                 eval_points=Default, scale_eval_points=Default,
                 label=Default, seed=Default, modulatory=Unconfigurable):
        super(Connection, self).__init__(label=label, seed=seed)

        self.pre = pre
        self.post = post

        self.synapse = synapse
        self.transform = transform
        self.scale_eval_points = scale_eval_points
        self.eval_points = eval_points  # Must be set before function
        self.function_info = function  # Must be set after transform
        self.solver = solver  # Must be set before learning rule
        self.learning_rule_type = learning_rule_type  # set after transform
        self.modulatory = modulatory

    @property
    def function(self):
        return self.function_info.function

    @function.setter
    def function(self, function):
        self.function_info = function

    @property
    def probeable(self):
        return ['output', 'input', 'weights']

    @property
    def pre_obj(self):
        return self.pre.obj if isinstance(self.pre, ObjView) else self.pre

    @property
    def pre_slice(self):
        return self.pre.slice if isinstance(self.pre, ObjView) else slice(None)

    @property
    def post_obj(self):
        return self.post.obj if isinstance(self.post, ObjView) else self.post

    @property
    def post_slice(self):
        return (self.post.slice if isinstance(self.post, ObjView)
                else slice(None))

    @property
    def size_in(self):
        """Output size of sliced `pre`; input size of the function."""
        return self.pre.size_out

    @property
    def size_mid(self):
        """Output size of the function; input size of the transform.

        If the function is None, then `size_in == size_mid`.
        """
        size = self.function_info.size
        return self.size_in if size is None else size

    @property
    def size_out(self):
        """Output size of the transform; input size to the sliced post."""
        return self.post.size_in

    @property
    def _str(self):
        if self.label is not None:
            return self.label

        desc = "" if self.function is None else " computing '%s'" % (
            getattr(self.function, '__name__', str(self.function)))
        return "from %s to %s%s" % (self.pre, self.post, desc)

    def __str__(self):
        return "<Connection %s>" % self._str

    def __repr__(self):
        return "<Connection at 0x%x %s>" % (id(self), self._str)

    @property
    def learning_rule(self):
        if self.learning_rule_type is not None and self._learning_rule is None:
            types = self.learning_rule_type
            if isinstance(types, dict):
                self._learning_rule = type(types)()  # dict of same type
                for k, v in iteritems(types):
                    self._learning_rule[k] = LearningRule(self, v)
            elif is_iterable(types):
                self._learning_rule = [LearningRule(self, v) for v in types]
            elif isinstance(types, LearningRuleType):
                self._learning_rule = LearningRule(self, types)
            else:
                raise ValidationError(
                    "Invalid type %r" % type(types).__name__,
                    attr='learning_rule_type', obj=self)

        return self._learning_rule

    @property
    def is_decoded(self):
        return not (self.solver.weights or (
            isinstance(self.pre_obj, Neurons) and
            isinstance(self.post_obj, Neurons)))


class LearningRule(object):
    def __init__(self, connection, learning_rule_type):
        self._connection = weakref.ref(connection)
        self.learning_rule_type = learning_rule_type

    def __repr__(self):
        return "<LearningRule at 0x%x modifying %r with type %r>" % (
            id(self), self.connection, self.learning_rule_type)

    def __str__(self):
        return "<LearningRule modifying %s with type %s>" % (
            self.connection, self.learning_rule_type)

    @property
    def connection(self):
        return self._connection()

    @property
    def error_type(self):
        return self.learning_rule_type.error_type

    @property
    def modifies(self):
        return self.learning_rule_type.modifies

    @property
    def probeable(self):
        return self.learning_rule_type.probeable

    @property
    def size_in(self):  # size of error signal
        if self.error_type == 'none':
            return 0
        elif self.error_type == 'scalar':
            return 1
        elif self.error_type == 'decoded':
            return (self.connection.post_obj.ensemble.size_in
                    if isinstance(self.connection.post_obj, Neurons) else
                    self.connection.size_out)
        elif self.error_type == 'neuron':
            raise NotImplementedError()
        else:
            raise ValidationError(
                "Unrecognized error type %r" % self.error_type,
                attr='error_type', obj=self)

    @property
    def size_out(self):
        return 0  # since a learning rule can't connect to anything
        # TODO: allow probing individual learning rules
