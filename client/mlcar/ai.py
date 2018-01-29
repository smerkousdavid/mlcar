from tensorforce.agents import PPOAgent
from mlcar.neuralnet.network import NeuralNetwork
from mlcar.neuralnet.genetic import *
from mlcar.configs import linear_map
from mlcar.logger import Logger
from datetime import datetime
from os.path import splitext

min_car_speed = 0.1
max_car_speed = 0.135
save_model_location = "/home/smerkous/Documents/mlcar/"

log = Logger("dispai")

network_spec = [
    dict(type="dense", size=2, activation="tanh"),
]

breeds = [
    TwinBreed(0),  # Create a copy of the best network and mutate it by 1%
    TwinBreed(1, 0.01),  # Create a copy of the second best network and mutate it by %10
    ChampionBreed(0, 1),  # Breed the top two networks (Do not mutate the networks)
    ChampionBreed(1, 2, 0.05),  # Breed the next two networks and mutate them by 5%
    ChampionBreed(0, 2, 0.02),  # Breed the first and third networks and mutate them by 2%
    ChampionBreed(2, 3, 0.03),  # Breed the third and fourth networks and mutate them by 3%
    RandomChampionBreed(0.05),  # Randomly breed two networks and mutate them by 5%
    RandomChampionBreed(0.01),  # Randomly breed two networks and mutate them by 1%
    NewRandomBreed(),  # Create a brand new random neural network
    NewRandomBreed()  # Same as above (It's good practice to have at least two random nns at the end of every generation
]


def build_agent():
    return PPOAgent(
        states_spec=dict(type="float", shape=(2,)),
        actions_spec=dict(continuous=True, type="float", shape=(2,),
                          min_value=min_car_speed, max_value=max_car_speed),
        network_spec=network_spec,
        batch_size=24,
        keep_last_timestep=True,
        step_optimizer=dict(
            type="adam",
            learning_rate=0.1
        ),
        optimization_steps=7,
        scope="ppo",
        discount=0.99,
        distributions_spec=None,
        entropy_regularization=0.01,
        baseline_mode=None,
        baseline=None,
        baseline_optimizer=None,
        gae_lambda=None,
        likelihood_ratio_clipping=0.15,
        summary_spec=None,
        distributed_spec=None
    )


class AI(object):
    def __init__(self):
        log.info("Building a new evolution")
        self._agent = build_agent()
        # self._n = NeuralNetwork(2, [2], 2)
        # self._n.generate()

        # self._e = Evolution(breeds, self._n)

    def no_lane(self):
        # self._e.remove_fitness(1)
        self._agent.observe(False, -0.5)

    def calc_reward(self, current):
        if abs(current) < 0.1:
            return 2
        return (-3 * pow(current, 2)) + 2

    def run(self, current, future):
        # predictions = self._e.run(current, future)
        # self._e.add_fitness()
        # return
        action = self._agent.act([current, future])
        self._agent.observe(False, self.calc_reward(current))  # First argument is termination
        action[1] = linear_map(action[1], min_car_speed, max_car_speed, -1, 1)
        return action

    def save(self):
        new_save_location = "%s%s" % (save_model_location, datetime.now().strftime("mlcar-%B-%d-%Y|%I:%M%p"))
        # self._e.save(new_save_location)
        self._agent.save_model(new_save_location, True)
        log.info("Saved model to %s" % new_save_location)

    def load(self, load_path):
        self._agent.restore_model(file=splitext(load_path)[0])
        # self._e.load(load_path, load_gen_meta=False)
        log.info("Loaded model from %s" % splitext(load_path)[0])

        # def next_breed(self):
        #    self._e.next_breed()
