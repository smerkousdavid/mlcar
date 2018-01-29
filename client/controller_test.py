from mlcar.controller import Controller
from mlcar.configs import load_configs, get_configs
from mlcar.logger import Logger
from time import sleep

log = Logger("tstcon")

if __name__ == "__main__":
    log.info("Testing the controller")

    load_configs()
    cont = Controller(get_configs())

    print("Turning right...")
    i = 0.0
    while i < 1.0:
        cont.drive(0, i)
        sleep(0.25)
        i += 0.1
    print("Done!")

    sleep(1)

    print("Turning left...")
    i = 1.0
    while i > -1.0:
        cont.drive(0, i)
        sleep(0.25)
        i -= 0.1

    print("Done!")
