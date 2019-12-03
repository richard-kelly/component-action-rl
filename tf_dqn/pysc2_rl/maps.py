from pysc2.maps import lib
import os

# get the absolute path to the maps
full_path_to_this_file = os.path.realpath(__file__)
path_to_project, _ = os.path.split(full_path_to_this_file)
path_to_maps = os.path.join(path_to_project, "maps", "combat_scenarios")


class CombatMaps(lib.Map):
    directory = path_to_maps
    download = "https://example.com"
    players = 1
    score_index = 0
    game_steps_per_episode = 0
    step_mul = 8


# get the list of maps and split on '.' because we don't want the .SC2Map at the end
full_map_names = os.listdir(path_to_maps)
mini_games = [x.split('.')[0] for x in full_map_names]

for name in mini_games:
    globals()[name] = type(name, (CombatMaps,), dict(filename=name))
