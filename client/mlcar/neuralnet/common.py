# -*- coding: utf-8 -*-
"""
File: common.py
Author: David Smerkous
Date: Tue Nov 21 2017

The neuralnet common functions
"""
from mlcar.neuralnet.errors import *
from StringIO import StringIO
import numpy as np


def linear_map(x, i_min, i_max, o_min, o_max):
    return ((x - i_min) * (o_max - o_min)) / ((i_max - i_min) + o_min)


def scalar_map(x, i_min, i_max):
    return (x - i_min) / (i_max - i_min)


def create_scalar_function(i_min, i_max):
    scaling = float(i_max - i_min)
    return lambda x: (x - i_min) / scaling


def sigmoid(x):
    return np.divide(1.0, np.add(1.0, np.exp(np.negative(x))))


def sigmoid_deriv(x):
    return np.multiply(x, np.subtract(x, 1.0))


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return np.subtract(1.0, np.power(x, 2.0))


def identity(x):
    return np.copy(x)


def identity_deriv(x):
    return np.ones(x.shape)


def binary(x):
    return np.where(x >= 0.0, 1.0, 0.0)


def binary_deriv(x):
    return np.zeros(x.shape)


def softsign(x):
    return np.divide(x, np.add(1.0, np.abs(x)))


def softsign_deriv(x):
    return np.divide(1.0, np.power(np.add(1.0, np.abs(x)), 2.0))


def relu(x):
    return np.where(x < 0.0, 0.0, x)


def relu_deriv(x):
    return np.where(x >= 0.0, 1.0, 0)


activation_functions = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "identity": identity,
    "binary": binary,
    "softsign": softsign,
    "relu": relu,
}

activation_deriv_functions = {
    "sigmoid": sigmoid_deriv,
    "tanh": tanh_deriv,
    "identity": identity_deriv,
    "binary": binary_deriv,
    "softsign": softsign_deriv,
    "relu": relu_deriv,
}


def activation(a_type="tanh"):
    global activation_functions
    try:
        return activation_functions[a_type]
    except KeyError:
        raise ActivationError("Activation function %s doesn't exist" % a_type)


def activation_deriv(a_type="tanh"):
    global activation_deriv_functions
    try:
        return activation_deriv_functions[a_type]
    except KeyError:
        raise ActivationError("Activation derivitave function %s doesn't exist" % a_type)


def calc_diff(p, c):
    if c == p:
        return 1.0
    try:
        return abs(float(c - p)) / p
    except ZeroDivisionError:
        return 0.0


def mx_to_str(arr):
    if type(arr).__module__ != np.__name__:
        arr = np.array(arr)
    f = StringIO()
    np.save(f, arr, allow_pickle=False)  # Python version compatible
    f.seek(0)
    mx_str = f.read()
    f.close()
    return mx_str


def str_to_mx(arr_str):
    f = StringIO(arr_str)
    f.seek(0)
    mx = np.load(f, allow_pickle=False)
    f.close()
    return mx


def average_breed(w, w2, **kwargs):
    """ Breed two weights by averaging both of them

    :param w: The first neural network's weights
    :param w2: The second neural network's weights
    :return: The average weights of the the two network weights

    :note
        This function can only average weights of the same size. Do not call this function on different layers
    """
    return np.divide(np.add(w, w2), 2.0)


def prefer_average_breed(w, w2, **kwargs):
    """ Breed two weights by averaging both of them (favoring the one with the higher fitness/rewards)

    :param w: The first neural network's weights
    :param w2: The second neural network's weights
    :param kwargs: f_fitness (first nn fitness) and s_fitness (second nn fitness)
    :return: The average weights of the the two network weights

    :note
        This function can only average weights of the same size. Do not call this function on different layers
    """
