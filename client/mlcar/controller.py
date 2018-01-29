from mlcar.logger import Logger
from serial import Serial, SerialException
from time import sleep

log = Logger("contrl")


class Controller(object):
    def __init__(self, configs):
        self._port = configs["ser_port"]
        log.info("Trying to connect to the controller...")
        while True:
            try:
                self._ser = Serial(self._port, configs["ser_baud"], timeout=1)
                break
            except SerialException:
                log.info("Failed to connect to controller! Retrying in 1 second...")
                sleep(1)
        log.info("Connected to the controller!")
        sleep(1)  # Wait one second for the controller to reset

    def drive(self, forward, turn):
        self._ser.write("S{}T{}E".format(int(forward * 100.0), int(turn * 100.0)))

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            log.info("Closing {}".format(self._port))
            self._ser.close()
            log.info("Closed!")
        except Exception as err:
            log.error("Failed to close {}! (err: {})".format(self._port, err))
