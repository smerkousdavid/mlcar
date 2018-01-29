# -*- coding: utf-8 -*-
"""
File: abstract.py
Author: David Smerkous
Date: Tue Nov 21 2017

The neural network abstraction layer
"""

# Local package imports
import mlcar.neuralnet.errors as errors
import mlcar.neuralnet.common as common
from mlcar.neuralnet.network import NeuralNetwork


class AbstractNeuralNetwork(NeuralNetwork):
    """ The general purpose neural network abstraction object

    :attr
        _attached_inputs (list: dict) The abstracted inputs
        _attached_outputs (list: dict) The abstracted outputs
        _clean_outputs (list: tuple) The "cleaned" (generated) abstracted outputs
        _hidden_layers (list: obj) A reference of hidden layers and their counts (if None then it's auto-generated)
        _act (obj: ref) The activation function (string or object reference)
        _act_d (obj: ref) The activation derivative (gradient descent) function (string or object reference)
    """
    def __init__(self, hl_n=None, act="tanh", act_d=None):
        """
        :param hl_n: The hidden layer count (either a list/tuple of node sizes, single node count, or list of HiddenLayer)
        :param act: The activation function reference (either a string or function reference to a custom activation function)
        :param act_d: The activation derivative function reference (same as above)

        :note
            Available activation functions are ("tanh", "sigmoid", "relu", "binary", "identity", "softsign")

        :example
            AbstractNeuralNetwork() # AbstractNeuralNetwork that automatically calculates the hidden layers

            # Custom hidden layers
            AbstractNeuralNetwork([2, 3]) # AbstractNeuralNetwork with two hidden layers one two nodes and the other three nodes

            def customact(x):
                return np.power(x, 2)

            def customact_deriv(x):
                return np.multiply(2, x)

            AbstractNueralNetwork(2, act=customact, act_d=customact_deriv) # Pass in your own activation functions


        """
        super(AbstractNeuralNetwork, self).__init__(1, 1, 1)
        self._attached_inputs = []
        self._attached_outputs = []
        self._clean_inputs = []
        self._clean_outputs = []
        self._hidden_layers = hl_n
        self._act = act
        self._act_d = act_d

    @staticmethod
    def _real_count(l):
        """ Private method to handle total input/output count for the attached methods

        :param l: The input attached list
        :return: The total count of unique names in the list
        """
        count = 0
        found = []
        for a in l:
            if not a["name"] in found:
                found.append(a["name"])
                count += 1
        return count

    def _calc_hidden(self):
        """ Private method to automatically calculate the complexity of the neural network

        :return: A list of the hidden layers and their node count

        :note
            This is obviously not going to be perfect and it's just designed to help beginners dive into
            neural networks. It's supposed to help add a dimension of complexity to properly fit (not underfit) the data
        """

        # Calculate the complexity of the network given the inputs
        complexity = 0
        for a in self._attached_inputs:
            if a["type"] is bool:
                complexity += 0.5
            else:
                complexity += 1.0

        # Add the output complexity (how many combinations are there available?)
        complexity += 1.5 * AbstractNeuralNetwork._real_count(self._attached_outputs)

        # Calculate the min and max nodes (There should be more hidden nodes than there are input nodes or data
        # or data-underfeeding may occur (it becomes too abstract)
        min_nodes = AbstractNeuralNetwork._real_count(self._attached_inputs) + 1
        max_nodes = int(2.0 * min_nodes) - 1

        # Calculate the amount of layers required
        layer_count = int(complexity / 3.0)
        if layer_count <= 1:
            n = [int((min_nodes + max_nodes) / 2.25)]
            if n[0] < 1:
                n[0] = 1  # Make sure there's at least one node
            return n
        elif layer_count == 2:
            n = [int((min_nodes + max_nodes) / 2.25), min_nodes]
            if n[0] < 1:
                n[0] = 1  # Make sure there's at least one node
            return n

        # If there are more than three layers (high complexity), then try to "peak" out the hidden layers
        # Example:
        #        - H -
        #       /  |  \
        #    - H - H - H -
        # I /  |   |   |  \ O
        # I -- H - H - H -- |
        # I \  |   |   |  / O
        #    - H - H - H -
        #       \  |  /
        #        - H -
        # This way there's an even distribution of min and max nodes and no data abstraction
        layers = []
        peak_value = int(layer_count / 2.0)
        for i in range(peak_value):
            layers.append(common.linear_map(i, 0, peak_value, min_nodes, max_nodes))
        for i in range(peak_value, -1, -1):
            layers.append(common.linear_map(i, peak_value, 0, max_nodes, min_nodes))
        return layers

    def add_input(self, name, input_type, min_value=None, max_value=None):
        """ Add an input to the neural network

        :param name: The name of the input (used for overwriting other inputs)
        :param input_type: The primitive input type (int, float, bool, etc...)
        :param min_value: If the type is a number then what's the lowest value that number can be
        :param max_value: If the type is a number then what's the largest value that number can be
        :return:
        """

        # Handle each type differently
        if input_type is str:
            raise errors.NotYetImplemented("Strings are currently not supported")
        elif input_type in (int, float, long):
            if min_value is None or max_value is None:
                raise errors.AbstractionError("min_value and max_value must be defined for numbers")
        elif input_type is bool:
            min_value = 0.0
            max_value = 1.0
        else:
            raise errors.AbstractionError("Unknown type is not supported")

        # Add the attached input to the neural network
        self._attached_inputs.append({
            "name": name,
            "type": input_type,
            "scalar": common.create_scalar_function(min_value, max_value)
        })

    def add_action(self, name, callback, min_thresh=0.25, max_thresh=1.0):
        """ Add an output to the neural network

        :param name: Namespace of the action (each namespace can have an infinite (as much as memory allows) amount of actions
        :param callback: The attached method reference for the action
        :param min_thresh: The minimum threshold value to trigger the action (default: between -1 and 1 (depending on the activation function)
        :param max_thresh: The maximum threshold value to trigger the action (default: between -1 and 1 (depending on the activation function)
        :return:

        :example
            def jump():
                print("JUMP")

            def high_jump():
                print("JUMP")

            def duck():
                print("DUCK")

            # All of these actions below are attached to one output on the neural network (all at different thresholds)
            n.add_action("jump", jump, min_thresh=0.1, max_thresh=0.5)
            n.add_action("jump", high_jump, min_thresh=0.5, max_thresh=1.0)
            n.add_action("jump", duck, min_thresh=-1.0, max_thresh=-0.1)
        """
        # Add the attached action to the neural network
        self._attached_outputs.append({
            "name": name,
            "callback": callback,
            "min_max": (min_thresh, max_thresh)
        })

    def create(self):
        """ Creates the neural network with randomly generated nodes

        :return:
        """
        # Calculate the hidden layers automatically if non have been specified
        if self._hidden_layers is None:
            self._hidden_layers = self._calc_hidden()

        in_count = AbstractNeuralNetwork._real_count(self._attached_inputs)
        out_count = AbstractNeuralNetwork._real_count(self._attached_outputs)

        if in_count < 1:
            raise errors.NeuralNetworkError("There must be at least one input!")

        if out_count < 1:
            raise errors.NeuralNetworkError("There must be at least one output!")

        # Create and generate the new neural network
        super(AbstractNeuralNetwork, self).__init__(
            in_count,
            self._hidden_layers,
            out_count,
            self._act,
            self._act_d)

        # Generate the random nodes
        self.generate()

        # Clean up the attached inputs into a smaller and more usable form
        self._clean_inputs = {}  # Clean up the clean outputs
        used_inputs = []  # Store the already used inputs in a temporary list
        t_a = 0  # The total output count
        for o in self._attached_inputs:
            if o["name"] in used_inputs:
                # Add an element override to the input
                c_ind = used_inputs.index(o["name"])
                self._clean_inputs[c_ind] = (o["type"], o["scalar"])
            else:
                # Add an action to the specified input element
                self._clean_inputs[t_a] = (o["type"], o["scalar"])
                t_a += 1
            used_inputs.append(o["name"])

        # Clean up the attached outputs into a smaller and more usable form
        self._clean_outputs = {}  # Clean up the clean outputs
        used_outputs = []  # Store the already used outputs in a temporary list
        t_a = 0  # The total output count
        for o in self._attached_outputs:
            if o["name"] in used_outputs:
                # Add an element override to the output
                c_ind = used_outputs.index(o["name"])
                self._clean_outputs[c_ind].append((o["callback"], o["min_max"], o["name"]))
            else:
                # Add an action to the specified output element
                self._clean_outputs[t_a] = [(o["callback"], o["min_max"], o["name"])]
                t_a += 1
            used_outputs.append(o["name"])

    def run(self, *inputs):
        """ Neural network prediction abstraction wrapper

        :param inputs: The values to predict
        :return: The raw predicted values

        :note
            The return can be used for anything the run method is designed to call the actions that were generated
            by the create method
        """

        # Scale and fix the inputs
        try:
            scaled_inputs = [float(self._clean_inputs[i][1](float(inputs[i]))) for i in xrange(len(inputs))]
        except Exception as err:
            raise errors.AbstractionError("Failed to clean the current inputs %s" % str(err))

        prediction = self.predict(scaled_inputs)  # Call the superclass prediction method
        actions_called = []
        try:
            for i in xrange(len(prediction)):  # Loop through each output
                c_p = prediction[i]  # Get the current prediction value
                # Loop through each action and check to see if the current prediction meets the threshold requirements
                for p in self._clean_outputs[i]:
                    if p[1][0] < c_p <= p[1][1]:
                        p[0]()  # Call the callback function
                        actions_called.append(p[2])  # Add a report of all the actions that were called
        except Exception as err:
            raise errors.PredictionError("Failed to run the neural network abstraction %s" % str(err))
        return prediction, actions_called
