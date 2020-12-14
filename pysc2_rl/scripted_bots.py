import numpy as np
import random
import math

from common import utils


def get_num_options_per_function(config):
    screen_size = config['env']['screen_size']
    minimap_size = config['env']['minimap_size']
    # this is hopefully the only place this has to be hard coded
    return dict(
        function=len(config['env']['computed_action_list']),
        screen=screen_size ** 2,
        minimap=minimap_size ** 2,
        screen2=screen_size ** 2,
        queued=2,
        control_group_act=5,
        control_group_id=config['env']['num_control_groups'],
        select_point_act=4,
        select_add=2,
        select_unit_act=4,
        select_unit_id=500,
        select_worker=4,
        build_queue_id=10,
        unload_id=500
    )


class NoopBot:
    def __init__(self):
        return

    def act(self, state, available_actions, eval_episode=False):
        return {'function': 0}

    def observe(self, terminal=False, reward=0, win=0, eval_episode=False):
        return


class RandomBot:
    def __init__(self, config):
        # these are computed at runtime (in pysc2_runner.py), and not manually set in the config
        self._action_components = config['env']['computed_action_components']
        self._action_list = config['env']['computed_action_list']
        self._num_control_groups = config['env']['num_control_groups']
        self._num_options = get_num_options_per_function(config)

    def act(self, state, available_actions, eval_episode=False):
        action = {}
        for name, using in self._action_components.items():
            if using:
                if name == 'function':
                    valid = np.in1d(self._action_list, available_actions)
                    try:
                        options = np.nonzero(valid)[0]
                        action[name] = np.random.choice(options)
                    except Exception:
                        print("WARNING: There were no valid actions. SOMETHING WENT WRONG.")
                        action[name] = available_actions[0]
                else:
                    action[name] = random.randint(0, self._num_options[name] - 1)
        return action

    def observe(self, terminal=False, reward=0, win=0, eval_episode=False):
        return


class AttackWeakestBot:
    def __init__(self, config):
        # these are computed at runtime (in pysc2_runner.py), and not manually set in the config
        self._action_components = config['env']['computed_action_components']
        self._action_list = config['env']['computed_action_list']
        self._num_control_groups = config['env']['num_control_groups']
        self._num_options = get_num_options_per_function(config)

        # pysc2 action id of attack_screen is 12
        self._attack_id = self._action_list.index(12)

    def act(self, state, available_actions, eval_episode=False):
        # if we can't attack do no_op
        if not state['available_actions'][self._attack_id]:
            return {'function': 0}

        action = {'function': self._attack_id}

        hp = state['screen_unit_hit_points']
        target = (0, 0)
        target_hp = 1000000
        # TODO: should do this in numpy; did it below in other bot, but not using this one anymore
        for x in range(hp.shape[0]):
            for y in range(hp.shape[1]):
                if state['screen_player_relative'][y][x] == 4:
                    if hp[y][x] < target_hp:
                        target_hp = hp[y][x]
                        target = (x, y)
        action['screen'] = utils.grid_to_flattened(hp.shape[0], target)

        return action

    def observe(self, terminal=False, reward=0, win=0, eval_episode=False):
        return


class AttackWeakestNearestBot:
    def __init__(self, config):
        # these are computed at runtime (in pysc2_runner.py), and not manually set in the config
        self._action_components = config['env']['computed_action_components']
        self._action_list = config['env']['computed_action_list']
        self._num_control_groups = config['env']['num_control_groups']
        self._num_options = get_num_options_per_function(config)

        # pysc2 action id of attack_screen is 12
        self._attack_id = self._action_list.index(12)

    def act(self, state, available_actions, eval_episode=False):
        # if we can't attack do no_op
        if not state['available_actions'][self._attack_id]:
            return {'function': 0}

        action = {'function': self._attack_id}

        rel = state['screen_player_relative']
        hp = state['screen_unit_hit_points']

        their_health = np.where(rel == 4, hp, np.NaN)
        result = np.where(their_health == np.nanmin(their_health))
        # zip the 2 arrays to get the exact coordinates
        list_of_cordinates = list(zip(result[1], result[0]))
        target = list_of_cordinates[0]
        if len(list_of_cordinates) > 1:
            friendly_unit_coords = np.where(rel == 1)
            y_avg = np.average(friendly_unit_coords[0])
            x_avg = np.average(friendly_unit_coords[1])

            dist = 1000000
            for x, y in list_of_cordinates:
                new_dist = math.sqrt((x_avg - x) ** 2 + (y_avg - y) ** 2)
                if new_dist < dist:
                    target = (x, y)
                    dist = new_dist

        action['screen'] = utils.grid_to_flattened(hp.shape[0], target)

        return action

    def observe(self, terminal=False, reward=0, win=0, eval_episode=False):
        return
