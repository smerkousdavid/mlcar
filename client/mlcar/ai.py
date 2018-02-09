from datetime import datetime
from os.path import splitext

from configs import linear_map
from logger import Logger
from neuralnet.genetic import *
from neuralnet.common import *
from neuralnet.network import *
from json import dump, load

# from tensorforce.agents import PPOAgent

min_car_speed = 0.1
# max_car_speed = 0.135
max_car_speed = 0.2
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
    return None
    """PPOAgent(
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
    )"""


class LinearGradDescent(object):
    def __init__(self, start_b=0.0, rate=0.00005):
        self._b = float(start_b)
        self._r = float(rate)
        self._epoch = float(1)

    def reset(self):
        self._epoch = float(1)

    def step(self, error):
        adj_err = -(2.0 / float(self._epoch)) * float(error)
        self._b -= (self._r * adj_err)
        self._epoch += 1
        print(self._b)

    def predict(self, v):
        return v * self._b

    def save(self):
        return {"b": self._b, "r": self._r, "e": self._epoch}

    def load(self, d):
        self._b = d["b"]
        self._r = d["r"]
        self._epoch = d["e"]


class AI(object):
    def __init__(self, is_linear=False):
        log.info("Building a new evolution")
        # self._agent = build_agent()
        self._l = is_linear
        if not is_linear:
            self._n = NeuralNetwork(2, [2], 2)
            self._n.generate()

            self._e = Evolution(breeds, self._n)
            self._e.create()
        else:
            self._p = LinearGradDescent(1, 0.20)  # Power gradient descent (20% learning rate)
            self._t = LinearGradDescent(0.65, 0.20)  # Turning gradient descent (20% learning rate)

    def no_lane(self):
        pass
        # self._e.remove_fitness(1)
        # self._agent.observe(False, -0.5)

    def calc_reward(self, current):
        if abs(current) < 0.1:
            return 2
        return (-3 * pow(current, 2)) + 2

    def run(self, current, future):
        if self._l:
            action = [0.075, #linear_map(abs(self._p.predict(float(future*1.8 + current*0.2) / 2.0)), 0, 0.9, 0.115, 0.09),
                      -self._t.predict((future + current) / 2.0)]
        else:
            action = self._e.run(current, future)
        # self._e.add_fitness(self.calc_reward(current))
        # return
        # action = self._agent.act([current, future])
        # self._agent.observe(False, self.calc_reward(current))  # First argument is termination
        # action[1] = linear_map(action[1], min_car_speed, max_car_speed, -1, 1)
        if action[0] > max_car_speed:
            action[0] = max_car_speed
        return action

    def save(self):
        new_save_location = "%s%s.weights" % (save_model_location, datetime.now().strftime("mlcar-%B-%d-%Y|%I:%M%p"))
        if self._l:
            data = {
                "p": self._p.save(),
                "t": self._t.save()
            }
            with open(new_save_location, 'w') as out_f:
                dump(data, out_f)
            self._p.save()
        else:
            self._e.save(new_save_location)
        # self._agent.save_model(new_save_location, True)
        log.info("Saved model to %s" % new_save_location)

    def load(self, load_path):
        # self._agent.restore_model(file=splitext(load_path)[0])
        if self._l:
            with open(load_path, 'r') as in_f:
                d = load(in_f)
                self._p.load(d["p"])
                self._t.load(d["t"])
        else:
            self._e.load(load_path, load_gen_meta=False)
        log.info("Loaded model from %s" % splitext(load_path)[0])

    def set_breed(self, breed):
        if not self._l:
            self._e.set_breed(breed)

    def set_fitness(self, fitness, completed, error):
        if self._l:
            self._p.step(error)
        else:
            self._e.set_fitness(fitness, completed)

    def breed_networks(self):
        if self._l:
            self._p.reset()
            self._t.reset()
        else:
            self._e.breed()

"""
a = LinearGradDescent(0, 0.1)
e = 0.1
for i in range(0, 40):
    err = e - a.predict(0.6)
    print("pred: %.2f err: %.2f" % (a.predict(0.6), err))
    a.step(err)
"""