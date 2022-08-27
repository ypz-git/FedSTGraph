from .fedavg import Client
from .fedbase import BasicServer


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)

    def sample(self):
        return [0]  # Client/Node ID
