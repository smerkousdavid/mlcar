""" 
File: errors.py
Author: David Smerkous
Date: Tue Nov 21 2017
The neural network error definitions
"""


class AbstractionError(Exception):
    """ Common abstraction errors
    Raised when the programmer decides to use an unknown type
    """

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class EvolutionError(AbstractionError):
    """ Common evolution errors
    Raised when the programmer decides to use an unknown type or fails to complete breeding methods
    """

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class PredictionError(AbstractionError):
    """ Common activation errors
    Raised when the prediction failed to load an action
    """

    def __init__(self, *args, **kwargs):
        AbstractionError.__init__(self, *args, **kwargs)


class NotYetImplemented(Exception):
    """ Not yet implemented error
    Raised when the requested feature hasn't yet been implemented
    """

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class BreedingError(Exception):
    """ Neural network breeding error
    Raised when the two neural networks couldn't breed against each other
    """

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class NeuralNetworkError(Exception):
    """ Common neural network errors
    Raised when the inputs for the neural network are incorrect
    """

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class ActivationError(Exception):
    """ Common activation errors
    Raised when the activation function isn't found
    """

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
