from pysc2.maps import lib


class MeleeMaps(lib.Map):
    directory = "rick_mini_games"
    download = "https://example.com"
    players = 1
    score_index = 0
    game_steps_per_episode = 0
    step_mul = 8


mini_games = [
    "8m_separated",
    "8m_separated_LTD_on_death"
]


for name in mini_games:
    globals()[name] = type(name, (MeleeMaps,), dict(filename=name))
