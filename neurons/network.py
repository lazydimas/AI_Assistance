import threading
import time
import asyncio
from .neuron import Neuron

class Network:
    def __init__(self):
        self.neurons = {}
        self.run = False

    def link(self, n1: Neuron, n2: Neuron):
        if n1 not in self.neurons:
            self.neurons[n1] = []
        if n2 not in self.neurons:
            self.neurons[n2] = []
        self.neurons[n1].append(n2)

    def add(self, neuron: Neuron):
        if neuron not in self.neurons:
            self.neurons[neuron] = []

    def maincycle(self):
        while self.run:
            for neuron in self.neurons:
                neuron.step()
                for connected in self.neurons[neuron]:
                    connected.step()
            time.sleep(0.5)

if __name__ == "__main__":
    network = Network()
    network.run = True
    thread = threading.Thread(target=network.maincycle)
    thread.start()
    n1 = Neuron("n1")
    n2 = Neuron("n2")
    network.add(n1)
    network.add(n2)
    network.link(n1, n2) 
