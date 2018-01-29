from __future__ import unicode_literals
from mlcar.neuralnet.network import NeuralNetwork
from mlcar.neuralnet.abstract import AbstractNeuralNetwork
from mlcar.neuralnet.errors import EvolutionError, BreedingError
from mlcar.neuralnet.common import average_breed
from copy import deepcopy as cpdata
from random import randint
from h5py import File
import six

if six.PY2:
    range = xrange


class BaseBreed(object):
    def __init__(self, mutation=None, breed_method=average_breed):
        self._breed_method = None
        self._mutation = mutation
        self.set_breed_method(breed_method)

    def set_breed_method(self, breed_method):
        if isinstance(breed_method, six.string_types):
            pass
        self._breed_method = breed_method

    def set_mutation(self, mutation):
        self._mutation = mutation

    def is_mutating(self):
        return self._mutation is not None

    def get_mutation(self):
        return self._mutation

    def breed(self, fitness, networks, base_network, nn_type):
        return None, None


class NewRandomBreed(BaseBreed):
    def __init__(self):
        super(NewRandomBreed, self).__init__()

    def breed(self, fitness, networks, base_network, nn_type):
        if nn_type == 0:
            base_network.generate()
        elif nn_type == 1:
            base_network.create()
        return base_network


class TwinBreed(BaseBreed):
    def __init__(self, breed_index, mutation=None, min_fitness_thresh=None, min_fitness_thresh_obj=NewRandomBreed()):
        super(TwinBreed, self).__init__(mutation=mutation)
        self._breed_index = breed_index
        self._min_fitness_thresh = min_fitness_thresh
        self._min_fitness_thresh_obj = min_fitness_thresh_obj

    def breed(self, fitness, networks, base_network, nn_type):
        if self._min_fitness_thresh is not None:
            if fitness[self._breed_index] < self._min_fitness_thresh:
                return self._min_fitness_thresh_obj.breed(fitness, networks, base_network, nn_type)
        twin_breed = cpdata(networks[self._breed_index])
        if self.is_mutating():
            twin_breed.mutate(self.get_mutation())
        return twin_breed


class ChampionBreed(BaseBreed):
    def __init__(self, first_breed_index, second_breed_index, mutation=None, breed_method=average_breed):
        super(ChampionBreed, self).__init__(mutation, breed_method)
        self._first_breed_index = first_breed_index
        self._secon_breed_index = second_breed_index

    def breed(self, fitness, networks, base_network, nn_type):
        baby_breed = networks[self._first_breed_index].breed(
            networks[self._secon_breed_index], breed_method=self._breed_method,
            f_fitness=fitness[self._first_breed_index],
            s_fitness=fitness[self._secon_breed_index])
        if self.is_mutating():
            baby_breed.mutate(self._mutation)
        return baby_breed


class RandomChampionBreed(BaseBreed):
    @staticmethod
    def _rand_ind(nets):
        return randint(0, len(nets) - 1)

    def breed(self, fitness, networks, base_network, nn_type):
        f_net = RandomChampionBreed._rand_ind(networks)
        s_net = RandomChampionBreed._rand_ind(networks)
        fails = 0
        while f_net == s_net and fails < 10:
            s_net = RandomChampionBreed._rand_ind(networks)
            fails += 1
        baby_breed = networks[f_net].breed(
            networks[s_net], breed_method=self._breed_method,
            f_fitness=fitness[f_net],
            s_fitness=fitness[s_net])
        if self.is_mutating():
            baby_breed.mutate(self._mutation)
        return baby_breed


class Evolution(object):
    def __init__(self, breeds, base_breed, save_gens=True, max_gens=-1, start_breed=0):
        if type(base_breed) is NeuralNetwork:
            self._nn_type = 0
        elif type(base_breed) is AbstractNeuralNetwork:
            self._nn_type = 1
        else:
            raise EvolutionError("The base breed must be a neural network")

        if not isinstance(breeds, (list, tuple)) or len(breeds) % 2 != 0:
            raise EvolutionError("The breeds list must be an even number")
        elif len(breeds) < 1:
            raise EvolutionError("There must be at least one breed")
        elif not isinstance(breeds[0], BaseBreed):
            raise EvolutionError("The breeding method is invalid")

        self._base_breed = cpdata(base_breed)
        self._max_gens = max_gens
        self._current_gen = 0
        self._save_gens = save_gens
        self._all_gens = []

        # Create the breeds
        self._breed_types = breeds
        self._breeds = [cpdata(self._base_breed) for _ in range(len(breeds))]
        self._fitness = [0] * len(breeds)
        self._current_breed = start_breed

    def create(self):
        for i in range(len(self._breeds)):
            if self._nn_type == 0:
                self._breeds[i].generate()
            elif self._nn_type == 1:
                self._breeds[i].create()
        self._fitness = [0] * len(self._breeds)

    def breed(self):

        # Save the current breeds
        if self._save_gens:
            self._all_gens.append((self._fitness, self._breeds))

        # Sort the champions
        networks = zip(*sorted(zip(self._fitness, self._breeds), key=lambda x: x[0], reverse=True))

        # Breed all of the breeds "you have to love that naming convention" with the predefined breed objects
        # NOTE TO SELF
        # ADD FAVORING fitness BREEDING METHOD
        for i in range(len(self._breed_types)):
            try:
                b_t = self._breed_types[i]
                baby_breed = b_t.breed(networks[0], networks[1], self._base_breed, self._nn_type)
                self._breeds[i] = baby_breed
            except Exception as err:
                raise BreedingError("Failed to breed index %d! %s" % (i, str(err)))

        # Reset the current breed data
        self._fitness = [0] * len(self._breeds)
        self._current_breed = 0

        # Increment the current generation counter
        self._current_gen += 1

        return networks

    def run(self, *inputs):
        if self._nn_type == 0:
            return self._breeds[self._current_breed].predict(inputs)
        elif self._nn_type == 1:
            return self._breeds[self._current_breed].run(*inputs)
        return self._breeds[self._current_breed]

    def set_fitness(self, fitness):
        self._fitness[self._current_breed] = fitness

    def add_fitness(self, fitness):
        self._fitness[self._current_breed] += fitness

    def remove_fitness(self, fitness):
        self._fitness[self._current_breed] -= fitness

    def get_fitness(self):
        return self._fitness[self._current_breed]

    def set_breed(self, ind):
        self._current_breed = ind

    def next_breed(self):
        self._current_breed += 1
        if self._current_breed == len(self._breeds):
            return {
                "gen_complete": True,
                "evolution_complete": self._current_gen >= self._max_gens,
                "breed": self._current_breed,
                "gen": self._current_gen
            }
        return {
            "gen_complete": False,
            "evolution_complete": self._current_gen >= self._max_gens,
            "breed": self._current_breed,
            "gen": self._current_gen
        }

    def get_breed(self):
        return self._current_breed

    def get_generation(self):
        return self._current_gen

    def get_generation_obj(self, generation_index):
        return self._all_gens[generation_index]

    def get_breed_at(self, generation_index, breed_index):
        return self.get_generation_obj(generation_index)[breed_index]

    def save(self, file_path=None):
        if file_path is None:
            file_path = "evolution.weights"
        h_file = File(file_path, 'w')

        # Metadata
        metadata = h_file.create_group("metadata")
        metadata.create_dataset("c_breed", data=[self._current_breed])
        metadata.create_dataset("m_gen", data=[self._max_gens])
        metadata.create_dataset("c_gen", data=[self._current_gen])
        metadata.create_dataset("save_gens", data=[self._save_gens])
        metadata.create_dataset("fitness", data=self._fitness)

        # Breeds
        breeds = h_file.create_group("breeds")
        for ind in range(len(self._breeds)):
            b_group = breeds.create_group("%d" % ind)
            self._breeds[ind].save(h5_group=b_group)

        h_file.close()

    def load(self, file_path=None, load_gen_meta=True):
        if file_path is None:
            file_path = "evolution.weights"

        h_file = File(file_path, 'r')

        # Metadata
        meta_data = h_file["metadata"]
        if load_gen_meta:
            self._current_breed = meta_data["c_breed"][0]
            self._max_gens = meta_data["m_gen"][0]
            self._current_gen = meta_data["c_gen"][0]
            self._save_gens = meta_data["save_gens"][0]
        self._fitness = meta_data["fitness"]

        # Breeds
        breeds = h_file["breeds"]
        for ind in range(len(breeds.keys())):
            nn = NeuralNetwork(1, 1, 1)
            nn.load(h5_group=breeds[str(ind)])
            self._breeds[ind] = nn

        h_file.close()
