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
        # handles when units try to move into the same space on the same frame:
        # 1 - cancel both; 2 - cancel random; 3 - cancel alternating
        self._move_conflict_resolution_strategy = utt['moveConflictResolutionStrategy']

        # a list of unit types, each of which is a dict with:
        #   ID (int)   : just equivalent to place in list, no relation to other uses of 'ID'
        #   name (str) : e.g. 'Resource'
        #   cost (int) : only meaningful for things that can be built
        #   hp (int)   : max hp
        #   minDamage (int), maxDamage (int), attackRange (int)
        #   produceTime (int) : The time it takes to produce this unit
        #   moveTime (int), attackTime (int), harvestTime (int), returnTime (int) : time cost to perform these actions
        #   harvestAmount (int) : only workers harvest and they harvest 1
        #   sightRadius (int) : for use in the partially observable ruleset
        #   isResource (bool), isStockpile (bool) [place a resource can be returned to - only Bases]
        #   canHarvest (bool), canMove (bool), canAttack (bool)
        #   produces (list of str): e.g. ['Base', 'Barracks']
        #   producedBy (list of str): e.g. ['Base']
        self._unit_types = utt['unitTypes']

    def _handle_pre_game_analysis(self, state, ms):
        # nothing to do with this for now
        # ms is the number of ms we have to exit this function (and send back a response, which happens elsewhere)
        # state is the staring state of the game (t=0)
        print('received pre game analysis')

    def _handle_get_action(self, state, player):
        state_for_rl = {}
        # state:
        #   map (0, 0) is top left
        game_frame = state['time']
        map_w = state['pgs']['width']
        map_h = state['pgs']['height']
        biggest = max([map_w, map_h])
        if biggest <= 8:
            map_size = 8
        elif biggest <= 16:
            map_size = 16
        elif biggest <= 32:
            map_size = 32
        elif biggest <= 64:
            map_size = 64
        elif biggest <= 128:
            map_size = 128
        else:
            print('Map dimension', biggest, 'is too big.')
            return

        # terrain can be 0 (empty) or 1 (wall); in row major order
        terrain = np.array([int(i) for i in state['pgs']['terrain']], dtype=np.int8).reshape((map_h, map_w))
        # for now pad non-square maps with 'walls' in the bottom and right
        terrain = np.pad(terrain, ((0, map_size - map_h), (0, map_size - map_w)), 'constant', constant_values=1)
        state_for_rl['terrain'] = terrain

        # list, each player is dict with ints "ID" and "resources"
        # the ID here is the same as player parameter to this function and "player" in unit object below
        players = state['pgs']['players']

        # list, each unit is dict with string "type" and
        # ints "ID", "player", "x", "y", "resources", "hitpoints"
        # ID here is a unique id for each unit in the game
        # resources is same for all bases belonging to a player, and equal to players[x]['resources']
        units = {}
        for unit in state['pgs']['units']:
            units[unit['ID']] = unit

        units_feature = np.zeros((map_size, map_size), dtype=np.int8)
        for _, unit in units.items():
            if unit['type'] == 'Base':
                units_feature[unit['y'], unit['x']] = 0
            elif unit['type'] == 'Barracks':
                units_feature[unit['y'], unit['x']] = 1
            elif unit['type'] == 'Worker':
                units_feature[unit['y'], unit['x']] = 2
            elif unit['type'] == 'Light':
                units_feature[unit['y'], unit['x']] = 3
            elif unit['type'] == 'Heavy':
                units_feature[unit['y'], unit['x']] = 4
            elif unit['type'] == 'Ranged':
                units_feature[unit['y'], unit['x']] = 5
            elif unit['type'] == 'Resource':
                units_feature[unit['y'], unit['x']] = 6
        state_for_rl['units'] = units_feature

        # TODO: try other representations of health... normalized real number? thermometer encoding?
        health_feature = np.zeros((map_size, map_size), dtype=np.int8)
        for _, unit in units.items():
            if unit['hitpoints'] == 1:
                health_feature[unit['y'], unit['x']] = 1
            elif unit['hitpoints'] == 2:
                health_feature[unit['y'], unit['x']] = 2
            elif unit['hitpoints'] == 3:
                health_feature[unit['y'], unit['x']] = 3
            elif unit['hitpoints'] == 4:
                health_feature[unit['y'], unit['x']] = 4
            elif unit['hitpoints'] >= 5:
                health_feature[unit['y'], unit['x']] = 5
        state_for_rl['health'] = health_feature

        # switch player ownership feature based on which player we are
        players_feature = np.zeros((map_size, map_size), dtype=np.int8)
        for _, unit in units.items():
            if unit['player'] == 0:
                if player == 0:
                    players_feature[unit['y'], unit['x']] = 1
                else:
                    players_feature[unit['y'], unit['x']] = 2
            elif unit['player'] == 1:
                if player == 0:
                    players_feature[unit['y'], unit['x']] = 2
                else:
                    players_feature[unit['y'], unit['x']] = 1
        state_for_rl['players'] = players_feature

        # actions ongoing for both players. A list with dicts:
        #   "ID":       int [unit ID],
        #   "time":     int [game frame when the action was given]
        #   "action":   {type, parameter, unitType, etc.]
        current_actions = {}
        for ongoing_action in state['actions']:
            current_actions[ongoing_action['ID']] = dict(time=ongoing_action['time'], action=ongoing_action['action'])
        # action_durations = {0: None, 1: }
        # for action in current_actions:

        eta_feature = np.zeros((map_size, map_size), dtype=np.int8)
        for ID, current_action in current_actions.items():
            action = current_action['action']
            time = current_action['time']
            unit = units[current_action['ID']]
            if action['action']['type'] == 1:
                # move
                eta =
                health_feature[unit['y'], unit['x']] = 1
        state_for_rl['eta'] = eta_feature

        # An action for a turn is a list with actions for each unit: a dict with
        # "unitID": int,
        #   "unitAction": {
        #       "type":      int [one of 0 (none/wait), 1 (move), 2 (harvest), 3 (return), 4 (produce), 5 (attack_location)],
        #       // used for direction (of move, harvest, return, produce) and duration (wait)
        #       "parameter": int [one of -1 (none), 0 (up), 1 (right), 2 (down), 3 (left) OR any positive int for wait],
        #       "x":         int [x coordinate of attack],
        #       "y":         int [y coordinate of attack],
        #       "unitType":  str [the name of the type of unit to produce with a produce action]
        #   }

        # Actions have to have a target / be legal at the time they are issued.
        # So even though actions take time to complete, there has to be a target at the time an attack is issued,
        # or a space has to be empty in order to issue a move action.
        # This is probably why attacks are all faster than moves.
        # some things we have to avoid in an action:
        #   don't send an action for a unit that is already doing an action
        #   don't try to produce two things in the same place

        actions = []

        for unit in units:
            if unit['player'] == player and unit['type'] == 'Worker' and unit['ID'] not in current_actions:
                action = dict(
                    unitID=unit['ID'],
                    unitAction=dict(
                        type=5,
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
