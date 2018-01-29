from nanomsg import Socket, BUS, NanoMsgAPIError, NanoMsgError
from threading import Thread
from time import sleep
from logger import Logger
from json import loads

log = Logger("comcat")


class Communicate(object):
    def __init__(self, configs):
        self._sock = Socket(BUS)
        self._configs = configs

    def connect(self):
        log.info("Connecting to %s" % self._configs["address"])
        self._sock.connect(self._configs["address"])

    def get(self):
        try:
            return loads(self._sock.recv())
        except Exception as err:
            if not "No JSON" in str(err):
                log.error("Failed pulling from car (err: %s)" % str(err))
            return None

    def __send_t(self, to_send):
        self._sock.send(to_send)
        sleep(0.01)
        self._sock.send(to_send)

    def send(self, *args):
        try:
            to_send = ""
            for arg in args:
                to_send += str(arg) + ";"
            Thread(target=self.__send_t, args=(to_send,)).start()

        except Exception as err:
            log.error("Failed sending to car (err: %s)" % str(err))

    def __exit__(self, exc_type, exc_val, exc_tb):
        log.info("Disconnecting from %s" % self._configs["address"])
        self._sock.close()
