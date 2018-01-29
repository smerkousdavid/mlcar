# -*- coding: utf-8 -*-
"""
File: network.py
Author: David Smerkous
Date: Tue Nov 21 2017

The neural network core
"""

# Local package imports
import mlcar.neuralnet.errors as errors
import mlcar.neuralnet.common as common

# Very few external depends
from copy import deepcopy as cpdata
from h5py import File
import numpy as np

whole_number_types = (int, long)


class Layer(object):
    """ Generic layer with nodes/weights

    :attr
        _node_count (int): The layer's input shape
        _weights (matrix: float): The layer's nodes
        _inputs (matrix: float): The feedforward core (training or prediction)
    """

    def __init__(self, node_count):
        """ Create a blank layer

        :param node_count: The amount of nodes the layer has
        """
        self._node_count = node_count
        self._weights = None
        self._inputs = None
        self._activation = common.tanh

    # SETTERS

    def set_activation(self, activation):
        """ Sets the layer's activation function

        :param activation: The activation function reference
        :return:
        """
        self._activation = activation

    def set_inputs(self, inputs):
        """ Set the layer's input matrix

        :param inputs: The layer's inputs
        :return:

        :note
            This function should only be called with the input layer as it will consume a lot of
            memory if the inputs are stored on every layer
        """
        self._inputs = np.array(inputs)

    def update_weights(self, data, delta):
        """ Back propagation

        :param data: The previous layer's prediction
        :param delta: The current layer's delta (direction)
        :return:

        :note
            Shift all high derives into a the corners (0 and 1)
            Fix the current weights based off of the delta magnitudes
        """
        self._weights += np.dot(data, delta)

    def set_weights(self, weights):
        """ Breed two nodes

        :param weights: weights (matrix: float): Override the current weights with a new set of weights/nodes
        :return:
        """
        self._weights = weights

    def generate_nodes(self, next_layer_count, ref=None):
        """ Random node creation

        :param next_layer_count: The amount of nodes the next layer has
        :param ref: The input node reference
        :return: The current layer's node count

        :note
            The ref should only be used on the output layer to match the
            last hidden layer's node count

            There's no need to create a node object as it takes a lot
            of overhead to reference and create (we will just use the
            weights as our "nodes") Lets return the output node count
        """
        node_count = ref if ref is not None else self._node_count
        self._weights = 2 * np.random.random((node_count, next_layer_count)) - 1
        return self._node_count

    # GETTERS

    def get_weights(self):
        """ Get the flipped/transformed weights

        :return: The current nodes in a numpy array that's flipped for the next
            layer (to make sure that outputs transpose to the next layer's inputs)

        :note
            This method should only be called to predict or train the
            current layer
        """
        return self._weights.T

    def get_node_count(self):
        """ Get the layer's node count

        :return: The current node count

        :note
            This is primarily used to feed the proceeding layer it's
            output count
        """
        return self._node_count

    def get_raw_weights(self):
        """ Get the current layer's raw weights

        :return: The layer's raw non-transformed weights

        :note
            This method does not transform the weights to be used on the
            proceeding layer. Use the standard get_weights to get the
            transformed weights
        """
        return self._weights

    def get_output(self, ref=None):
        """ Forward propagation

        Gets the dot product of the inputs and weights which then each
        node is sent through the activation function and then returned

        :param ref: The layer's refernce point
        :return: A matrix of the activated nodes
        """
        inputs = ref if ref is not None else self._inputs
        return self._activation(np.dot(inputs, self._weights))

    def pickle(self):
        if self._weights is None:
            raise errors.NeuralNetworkError("The layer cannot be pickled! (No weights)")

        if self._inputs is None:
            inputs = []
        else:
            inputs = self._inputs
        return np.array([str(self._node_count),
                         common.mx_to_str(self._weights),
                         common.mx_to_str(inputs)])

    def depickle(self, pickled):
        if pickled is None:
            raise errors.NeuralNetworkError("The layer could not be depicked! (Pickled is None)")
        self._node_count = int(pickled[0])
        self._weights = common.str_to_mx(pickled[1])
        self._inputs = common.str_to_mx(pickled[2])


class InputLayer(Layer):
    """ Input layer of the neural network

    :note
        The inputs are the values you want to train or predict

    """

    def __init__(self, node_count):
        Layer.__init__(self, node_count)

    def get_inputs(self):
        """ Only the input layer needs to capture its inputs

        :return: The values that the input layer wants to feed through
        """
        return self._inputs

    def update_weights(self, delta):
        """ Update the layer's weights

        :param delta: The current layer's delta
        :return:

        :note
            This function isn't necessary but it does make the code look
            a lot cleaner
        """
        Layer.update_weights(self, self.get_inputs().T, delta)


class HiddenLayer(Layer):
    """ Hidden layer of the neural network (the core of the network)

    :note
        This is a generic abstraction. It can be removed to reduce the
        overhead but then it becomes harder to read the code
    """

    def __init__(self, node_count):
        Layer.__init__(self, node_count)


class OutputLayer(Layer):
    """ Hidden layer of the neural network (the core of the network)

    :note
        This is a generic abstraction. It can be removed to reduce the
        overhead but then it becomes harder to read the code
    """

    def __init__(self, node_count):
        Layer.__init__(self, node_count)

    def generate_nodes(self, last_layer_count):
        """ Override the layer's default generate_nodes object

        :param last_layer_count: The previous layer's node count
        :return: The previous layer count

        :note
            This is the last layer so our output node count is equal to
            the total layer node count
        """
        Layer.generate_nodes(self, self._node_count, last_layer_count)
        return last_layer_count


class NeuralNetwork(object):
    """ The neural network core object

    :attr
        _input_l (InputLayer): The input layer of network
        _hidden_l (list: HiddenLayer): The hidden layers of the network
        _output_l (OutputLayer): The output layer of the network
        _expected (matrix: float): The expected output of the neural network
        _act_str: The activation function string reference (not the function reference)
        _act: The activation function function reference
        _act_d: The activation derivative function reference
    """

    def __init__(self, il_n, hl_n, ol_n, act="tanh", act_d=None):
        """

        :param il_n: The input layer (either the node count or the InputLayer object)
        :param hl_n: The hidden layer count (either a list/tuple of node sizes, single node count, or list of HiddenLayer)
        :param ol_n: The output layer (either the node count or the OutputLayer object)
        :param act: The activation function reference (either a string or function reference to a custom activation function)
        :param act_d: The activation derivative function reference (same as above)

        :note
            Available activation functions are ("tanh", "sigmoid", "relu", "binary", "identity", "softsign")

        :example
            NeuralNetwork(1, 3, 2) # NeuralNetwork with 1 input nodes 3 hidden layer nodes and 2 output nodes
            NeuralNetwork(3, [2, 3], 1) # NeuralNetwork with 3 input nodes two hidden layers (2 and 3 nodes) and one output node

            il = InputLayer(3) # Three input nodes
            hl = [ # Three hidden layers
                HiddenLayer(2), # Two nodes
                HiddenLayer(5), # Five nodes
                HiddenLayer(10) # Ten nodes
            ]
            ol = OutputLayer(2) # Two nodes
            NeuralNetwork(il, hl, ol) # Create the neural network object based off of the above layer requirements
            NeuralNetwork(InputLayer(2), [2, 2, 3], 3) # Mix and mash any of the above examples

            # Custom activation function examples
            NeuralNetwork(2, [1, 5], 3, act="sigmoid") # Use the prebuilt activation functions

            def customact(x):
                return np.power(x, 2)

            def customact_deriv(x):
                return np.multiply(2, x)

            NueralNetwork(2, [1, 5], 3, act=customact, act_d=customact_deriv) # Pass in your own


        """

        # Create the input layer
        if isinstance(il_n, whole_number_types):
            self._input_l = InputLayer(il_n)
        elif isinstance(il_n, InputLayer):
            self._input_l = cpdata(il_n)
        else:
            raise errors.NeuralNetworkError("Input layer argument is invalid")

        # Create the hidden layers
        if isinstance(hl_n, whole_number_types):
            if hl_n != 0:
                self._hidden_l = [HiddenLayer(hl_n)]
            else:
                self._hidden_l = []
        elif isinstance(hl_n, (list, tuple)):
            if len(hl_n) > 0:
                if isinstance(hl_n[0], HiddenLayer):
                    self._hidden_l = cpdata(hl_n)
                elif isinstance(hl_n[0], whole_number_types):
                    self._hidden_l = [HiddenLayer(h) for h in hl_n]
                else:
                    raise errors.NeuralNetworkError("Hidden layer list arguments are invalid")
            else:
                self._hidden_l = []
        else:
            raise errors.NeuralNetworkError("Hidden layer argument is invalid")

        # Create the output layer
        if isinstance(ol_n, whole_number_types):
            self._output_l = OutputLayer(ol_n)
        elif isinstance(ol_n, OutputLayer):
            self._output_l = cpdata(ol_n)
        else:
            raise errors.NeuralNetworkError("Output layer argument is invalid")

        # Update the layers' activation functions
        self._act_str = act
        self._act = None
        self._act_d = None
        self.set_activation(act, act_d)

        # Set the fit data to nothing
        self._expected = []

    def set_activation(self, act="tanh", act_d=None):
        """ Set all of the layer's activation functions

        :param act: The activation function reference (either a string or function reference to a custom activation function)
        :param act_d: The activation derivative function reference (same as above)
        :return:
        """
        # Pull the activation and activation derivative function reference
        if isinstance(act, basestring):
            self._act_str = act
            self._act = common.activation(act)
            self._act_d = common.activation_deriv(act)
        else:
            self._act = act

        # Pull the activation function derivative reference if it was provided
        if act_d is not None:
            self._act_d = act_d

        # Set the layer's activation functions
        self._input_l.set_activation(self._act)
        for i in range(len(self._hidden_l)):
            self._hidden_l[i].set_activation(self._act)
        self._output_l.set_activation(self._act)

    # SETTERS

    def set_inputs(self, *inputs):
        """ Set the neural network's InputLayer inputs

        :param inputs: The values to train or predict
        :return:
        """
        self._input_l.set_inputs(inputs)

    def set_expected(self, *expected):
        """ Set the neural network's expected output

        :param expected: The expected out (same column size as the inputs and row count as the OutputLayer)
        :return:
        """
        self._expected = np.array([[e] for e in expected])

    def get_raw_hidden_weights(self):
        """ Get the non-transformed hidden layer weights

        :return: A list of matrices that associate (in order) with each HiddenLayer weights
        """
        return [l.get_raw_weights() for l in self._hidden_l]

    def get_raw_weights(self):
        """ Get all of the raw non-transformed layer weights

        :return: A tuple of matrices that include all of layer weights
        """
        return self._input_l.get_raw_weights(), self._output_l.get_raw_weights(), self.get_raw_hidden_weights()

    def generate(self):
        """ Generate the random weights for all of the layers

        Connects the sizes of all the layers

        :return:

        :note
            This must be called every time the node count on a layer changes if it's not update the entire
            neural network will crash. Also, if the node count does change then the entire neural network must be
            rebuilt
        """
        # Pull the layer before
        if len(self._hidden_l) > 0:
            # Get the last available hidden layer node count
            n_count = self._hidden_l[-1].get_node_count()
        else:
            # Get the input layer's node count if there aren't any hidden layers
            n_count = self._input_l.get_node_count()
        n_count = self._output_l.generate_nodes(n_count)  # Get the output layer's node count

        # Loop through the available hidden layers
        for i in xrange(len(self._hidden_l) - 1, -1, -1):
            n_count = self._hidden_l[i].generate_nodes(n_count)  # Get the hidden layers' node counts

        self._input_l.generate_nodes(n_count)  # Generate the input layers' nodes

    @staticmethod
    def _gen_mutation(orig, perc):
        """ Static method to mutate weights

        :param orig: The original unaltered weights
        :param perc: The percentage change/mutation each value should receive
        :return: The mutated weights matrix (same size)

        :note
            The current method for mutating weights is as follows
            1. Capture the original weights size
            2. Generate a random matrix between -1 and 1 (mean 0) of the same size
            3. Multiply each value of the random matrix by a percentage change coefficient
            4. Add the original and change matrix together
        """
        return np.add(orig, np.multiply(perc, np.subtract(np.multiply(2.0, np.random.random(orig.shape)), 1.0)))

    def mutate(self, perc):
        """ Mutates all of the layers by the given percentage

        :param perc: A float between 0 and 1 representing a percentage
        :return:
        """
        # Get the current network weights
        self_weights = self.get_raw_weights()

        # Mutate the weights with pseudo-random values
        i_mutate = self._gen_mutation(self_weights[0], perc)
        o_mutate = self._gen_mutation(self_weights[1], perc)

        # Update the layer's weights
        self._input_l.set_weights(i_mutate)
        self._output_l.set_weights(o_mutate)

        # Mutate the hidden layers
        for i in range(len(self._hidden_l)):
            h_mutate = self._gen_mutation(self_weights[2][i], perc)
            self._hidden_l[i].set_weights(h_mutate)

    def breed(self, nn, breed_method=common.average_breed, **kwargs):
        """ Breeds the current network with the other to create a baby network

        :param nn: The other NeuralNetwork object to breed with
        :param breed_method: The function to breed the weights
        :return: A baby NeuralNetwork object
        """
        # Get all of the networks' weights
        nn_weights = nn.get_raw_weights()
        self_weights = self.get_raw_weights()

        # Make sure the layers' weights match up
        if nn_weights[0].shape != self_weights[0].shape:
            raise errors.BreedingError("Input layer mismatch")
        if nn_weights[1].shape != self_weights[1].shape:
            raise errors.BreedingError("Output layer mismatch")

        # Loop through and check all of the hidden layers' weights
        for w in zip(nn_weights[2], self_weights[2]):
            if w[0].shape != w[1].shape:
                raise errors.BreedingError("Hidden layer mismatch")

        # Average out the weights
        breed_method([], [])
        n_l_weights = breed_method(nn_weights[0], self_weights[0])
        n_o_weights = breed_method(nn_weights[1], self_weights[1], **kwargs)

        # Create a copy of the current network then cross-bread the above two networks
        new_network = cpdata(self)
        new_network.set_inputs()  # Clear the inputs
        new_network._input_l.set_weights(n_l_weights)
        new_network._output_l.set_weights(n_o_weights)

        h_l = []
        for i in range(len(new_network._hidden_l)):
            c_h = cpdata(new_network._hidden_l[i])
            c_h.set_weights(breed_method(nn_weights[2][i], nn_weights[2][i], **kwargs))
            h_l.append(c_h)

        return new_network

    def train(self):
        """ Fit the input data so that it can match the expected values
        :return: The average error of the network
        """
        # Feedforward
        i_out = self._input_l.get_output()
        # Copy the input data
        in_o = np.copy(i_out)
        m_train = []
        for i in xrange(len(self._hidden_l)):
            i_out = self._hidden_l[i].get_output(i_out)
            m_train.append(i_out)
        i_out = self._output_l.get_output(i_out)

        # Calculate the error
        o_error = np.subtract(self._expected, i_out)

        # Calculate the output delta
        delta = o_error * self._act_d(i_out)

        # Detect the direction of change
        f_out = m_train[-1] if len(m_train) > 0 else in_o

        # Copy the previous weights
        prev_weights = self._output_l.get_weights()

        # Update the output layer's weights
        self._output_l.update_weights(f_out.T, delta)

        # Loop back through the layers and update the weights
        if len(self._hidden_l) > 1:
            for i in xrange(len(m_train) - 1, 0, -1):
                # Get the error of the current layer
                error = delta.dot(prev_weights)

                # Get the error direction
                delta = error * self._act_d(m_train[i])

                # Copy the weights for the next layer
                prev_weights = self._hidden_l[i].get_weights()

                # Update the hidden layer's weights
                self._hidden_l[i].update_weights(m_train[i - 1].T, delta)

        # Update the weights of the last hidden layer if it exists
        if len(self._hidden_l) > 0:
            # Calculate the error
            error = delta.dot(prev_weights)

            # Get the error direction
            delta = error * self._act_d(m_train[0])

            # Copy the weights for the input layer
            prev_weights = self._hidden_l[0].get_weights()

            # Update the last hidden layer's weights
            self._hidden_l[0].update_weights(in_o.T, delta)

        # Calculate the input layer's error
        error = delta.dot(prev_weights)

        # Get the error direction
        delta = error * self._act_d(in_o)

        # Update the input layer's weights
        self._input_l.update_weights(delta)

        return np.mean(np.abs(o_error))

    def predict(self, values):
        """ Get the neural network's prediction

        :param values: The input values of the prediction
        :return: The expected result matrix (shape column count is similar to that of the OutputLayer)
        """
        i_out = self._input_l.get_output(np.array(values))
        for i in xrange(len(self._hidden_l)):
            i_out = self._hidden_l[i].get_output(i_out)
        i_out = self._output_l.get_output(i_out)
        return i_out

    def save(self, file_path=None, compression="gzip", compression_opts=9, h5_group=None):
        if file_path is None:
            file_path = "network.weights"

        if h5_group is None:
            h_file = File(file_path, 'w')
        else:
            h_file = h5_group

        # Metadata
        meta_data = h_file.create_group("metadata")
        meta_data.create_dataset("act", data=[self._act_str])
        meta_data.create_dataset("expected", data=self._expected)

        # Input Layer
        i_layer = h_file.create_group("i_layer")
        i_layer.create_dataset("i_data", data=self._input_l.pickle(),
                               compression=compression,
                               compression_opts=compression_opts)

        # Hidden Layers
        h_layers = h_file.create_group("h_layers")
        h_data = []
        for ind in range(len(self._hidden_l)):
            h_data.append(self._hidden_l[ind].pickle())
        h_layers.create_dataset("h_data", data=h_data,
                                compression=compression,
                                compression_opts=compression_opts)

        # Output Layer
        o_layer = h_file.create_group("o_layer")
        o_layer.create_dataset("o_data", data=self._output_l.pickle(),
                               compression=compression,
                               compression_opts=compression_opts)
        if h5_group is None:
            h_file.close()

    def load(self, file_path=None, h5_group=None):
        if file_path is None:
            file_path = "network.weights"

        if h5_group is None:
            h_file = File(file_path, 'r')
        else:
            h_file = h5_group

        # Metadata
        meta_data = h_file["metadata"]
        self._act_str = meta_data["act"][0]
        self._expected = meta_data["expected"]

        # Input Layer
        i_layer = h_file["i_layer"]
        self._input_l.depickle(i_layer["i_data"])

        # Hidden Layers
        h_layers = h_file["h_layers"]
        h_data = h_layers["h_data"]
        self._hidden_l = []
        for ind in range(len(h_data)):
            h_l = HiddenLayer(0)
            h_l.depickle(h_data[ind])
            self._hidden_l.append(h_l)

        # Output Layer
        o_layer = h_file["o_layer"]
        self._output_l.depickle(o_layer["o_data"])

        if h5_group is None:
            h_file.close()
