import numpy as np

def flattened_to_grid(elem):
    # x, y
    return elem % n, elem / n


def grid_to_flattened(tile):
    return tile[0] * n + tile[1]


def get_neighbours(x, y):
    neighbours = []
    if x > 0:
        neighbours.append((x - 1, y))
    if x < n - 1:
        neighbours.append((x + 1, y))
    if y > 0:
        neighbours.append((x, y - 1))
    if y < n - 1:
        neighbours.append((x, y + 1))
    return neighbours

# screen is 2x2
q_vals = dict(
    select=np.array([[3, 4, 5, 2], [1, 2, 3, 4]], dtype=np.float32),
    type=np.array([[3.4, 5.6, 7.8, 9.0, 1, 2], [3, 2, 3, 14, 2, 3]], dtype=np.float32),
    param=np.array([[3, 4, 5, 2], [1, 2, 3, 4]], dtype=np.float32),
    unit_type=np.array([[3.4, 5.6, 7.8, 9.0, 1, 2], [3, 2, 3, 14, 2, 3]], dtype=np.float32)
)

states = dict(
    player=np.array([[[0, 2], [1, 0]], [[0, 1], [0, 2]]], dtype=np.int8),
    eta=np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]], dtype=np.int8),
    units=np.array([[[0, 3], [3, 0]], [[0, 6], [0, 3]]], dtype=np.int8),
    available_resources=np.array([20, 20]), dtype=np.int32
)

n = 2
worker_cost = 1
light_cost = 2
heavy_cost = 3
ranged_cost = 2
barracks_cost = 5
base_cost = 10

batch = 2
actions = {}
for name, vals in q_vals.items():
    actions[name] = np.zeros(batch, dtype=np.int32)

# SELECT = unit that is doing the action
# mask out tiles that don't have our units
q_vals['select'] = np.where(states['player'].reshape((batch, n * n)) == 1, q_vals['select'], np.nan)
# mask out tiles with units that already have actions
q_vals['select'] = np.where(states['eta'].reshape((batch, n * n)) == 0, q_vals['select'], np.nan)
actions['select'] = np.nanargmax(q_vals['select'], axis=1)

# TYPE = type of action: [no-op, move, harvest, return, produce, attack]
# q_vals['type'] = np.where(np.isin(states['']))
unit_types = np.reshape(states['units'], (batch, n * n))[np.arange(batch), actions['select']]
for i in range(batch):
    if unit_types[i] == 0:
        # bases can only do no_op and produce (0 and 4)
        q_vals['type'][i][np.array([1, 2, 3, 5])] = np.nan
        # are there enough resources to produce?
        if states['available_resources'][i] > worker_cost:
            # is there a space to produce?
            flattened_coords = actions['select'][i]
            x, y = flattened_to_grid(flattened_coords)
            for neighbours in get_neighbours(x, y):
                if





