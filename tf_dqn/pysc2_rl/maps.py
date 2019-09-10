from pysc2.maps import lib


class CombatMaps(lib.Map):
    directory = "combat_scenarios"
    download = "https://example.com"
    players = 1
    score_index = 0
    game_steps_per_episode = 0
    step_mul = 8


mini_games = [
    "combat_8m_v_8m_sparse_reward",
    "combat_8m_v_8m_LTD_on_damage_norm_factor_100000",
    "combat_8m_v_8m_LTD_on_death_norm_factor_100000",
    "combat_8m_v_8m_LTD2_on_damage_norm_factor_100000",
    "combat_8m_v_8m_LTD2_on_death_norm_factor_100000"
]


for name in mini_games:
    globals()[name] = type(name, (CombatMaps,), dict(filename=name))
