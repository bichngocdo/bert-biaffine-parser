import re
from collections import OrderedDict

import tensorflow as tf


def get_assignment_map_from_checkpoint(checkpoint, scope=None):
    if scope is None:
        scope = ''
    elif not scope.endswith('/'):
        scope += '/'
    vars = tf.trainable_variables(scope)
    ckpt_vars = tf.train.list_variables(checkpoint)
    assignment_map = OrderedDict()
    initialized_variable_names = dict()

    name2var = OrderedDict()
    for var in vars:
        name = var.name
        m = re.match('^(.*):\\d+$', name)
        if m is not None:
            name = m.group(1)
        name2var[name] = var

    for name, _ in ckpt_vars:
        new_name = scope + name
        if new_name not in name2var:
            continue
        assignment_map[name] = new_name
        initialized_variable_names[new_name] = name
        initialized_variable_names[new_name + ':0'] = name

    return assignment_map, initialized_variable_names


def initialize_from_checkpoint(checkpoint):
    scope = tf.get_variable_scope().name
    assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(checkpoint, scope)
    tf.train.init_from_checkpoint(checkpoint, assignment_map)
