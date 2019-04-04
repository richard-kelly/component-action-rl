# Echo server program
import socket
import json
import re
import numpy as np
import random

HOST = '127.0.0.1'
PORT = 9898


class MRTSServer:
    def __init__(self):
        self._conn = None

        # game details
        self._budgets = None
        self._unit_types = None
        self._move_conflict_resolution_strategy = None

        self._next_message = None

    def _handle_unit_type_table(self, utt):
        self._move_conflict_resolution_strategy = utt['moveConflictResolutionStrategy']
        self._unit_types = utt['unitTypes']

    def _handle_pre_game_analysis(self, state, ms):
        # nothing to do with this for now
        print('received pre game analysis')

    def _handle_get_action(self, state, player):
        # state:
        #   map (0, 0) is top left
        game_frame = state['time']
        map_w = state['pgs']['width']
        map_h = state['pgs']['height']
        terrain = np.array([int(i) for i in state['pgs']['terrain']], dtype=np.int32).reshape((map_w, map_h))

        # list, each player is dict with ints "ID" and "resources"
        # the ID here is the same as player parameter to this function and "player" in unit object below
        players = state['pgs']['players']

        # list, each unit is dict with string "type" and
        # ints "ID", "player", "x", "y", "resources", "hitpoints"
        # ID here is a unique id for each unit in the game
        units = state['pgs']['units']

        # actions ongoing for both players. A list with dicts:
        #   "ID": int [unit ID],
        #   "time": int [game frame when the action was given]
        #   "action": {type, paramater, unitType, etc.]
        current_actions = state['actions']
        # action_durations = {0: None, 1: }
        # for action in current_actions:

        units_taking_actions = [elem['ID'] for elem in current_actions]

        # An action for a turn is a list with actions for each unit: a dict with
        # "unitID": int,
        #   "unitAction": {
        #       "type": int [one of 0 (none/wait), 1 (move), 2 (harvest), 3 (return), 4 (produce), 5 (attack_location)],
        #       // used for direction (of move, harvest, return, produce) and duration (wait)
        #       "parameter": int [one of -1 (none), 0 (up), 1 (right), 2 (down), 3 (left) OR any positive int for wait],
        #       "x": int [x coordinate of attack],
        #       "y": int [y coordinate of attack],
        #       "unitType": string [the name of the type of unit to produce with a produce action]
        #   }

        # no_op for every unit
        actions = []
        for unit in units:
            if unit['player'] == player and unit['type'] == 'Worker' and unit['ID'] not in units_taking_actions:
                action = dict(
                    unitID=unit['ID'],
                    unitAction=dict(
                        type=1,
                        # parameter=random.randint(0, 5)
                        parameter=2
                    )
                )
                actions.append(action)

        return json.dumps(actions)

    def _handle_game_over(self, winner):
        print('winner:', winner)

    def listen(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((HOST, PORT))
            sock.listen(1)
            self._conn, addr = sock.accept()
            with self._conn:
                print('Connected by', addr)
                # client expects acknowledgement of connection
                self._conn.sendall(b"ack\n")
                while True:
                    data = self._conn.recv(16384)
                    if not data:
                        break
                    decoded = data.decode('utf-8')
                    # decide what to do based on first word
                    if re.search("^budget", decoded):
                        self._budgets = [int(i) for i in decoded.split()[1:]]
                        self._conn.sendall(b"ack\n")
                    elif re.search("^utt", decoded):
                        self._handle_unit_type_table(json.loads(decoded.split('\n')[1]))
                        self._conn.sendall(b"ack\n")
                    elif re.search("^preGameAnalysis", decoded):
                        lines = decoded.split('\n')
                        ms = int(lines[0].split()[1])
                        self._handle_pre_game_analysis(json.loads(lines[1]), ms)
                        self._conn.sendall(b"ack\n")
                    elif re.search("^getAction", decoded):
                        lines = decoded.split('\n')
                        player = int(lines[0].split()[1])
                        action = self._handle_get_action(json.loads(lines[1]), player)
                        message = action + '\n'
                        self._conn.sendall(message.encode('utf-8'))
                    elif re.search("^gameOver", decoded):
                        self._handle_game_over(int(decoded.split()[1]))
                        self._conn.sendall(b"ack\n")
                    else:
                        print('Message type not recognized!')
                        print(decoded)


def main():
    server = MRTSServer()
    server.listen()


if __name__ == '__main__':
    main()
