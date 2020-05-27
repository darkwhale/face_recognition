import random
import time


def generate_key():
    return "{}{}".format(int(time.time() * 1000), random.randint(0, 90000) + 10000)
