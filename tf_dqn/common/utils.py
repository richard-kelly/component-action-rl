import numpy as np
import random
import math

def one_hot_encode_int_array(array, num_cats):
    shape = tuple(list(array.shape) + [num_cats])
    one_hot = np.zeros(shape)
    for i in range(num_cats):
        one_hot[..., i] = np.where(array == i, 1, 0)
    return one_hot


def one_hot_encode_int_arrays(*args):
    # This version doesn't use zero as a category, starts at 1
    # each arg is a tuple, (array, num_cats)
    # each array should have the same dimensions
    num_cats = 0
    for arg in args:
        num_cats += arg[1]

    shape = tuple(list(args[0][0].shape) + [num_cats])
    one_hot = np.zeros(shape)
    layer = 0
    for arg in args:
        for i in range(1, arg[1] + 1):
            one_hot[..., layer] = np.where(arg[0] == i, 1, 0)
            layer += 1
    return one_hot


def log_uniform(a, b, base=10):
    return a ** ((((math.log(b, base) / math.log(a, base)) - 1) * random.random()) + 1)


def main():
    print("testing one hot encode:")
    x = np.array([0, 0, 1, 2, 3, 0, 1, 2, 3]).reshape(3, 3)
    print('x:')
    print(x)
    print('one_hot_encode_int_array(x, 5)')
    new_array = one_hot_encode_int_array(x, 5)
    print(new_array)
    print(new_array.shape)
    for i in range(5):
        print(new_array[:, :, i])


if __name__ == "__main__":
    main()

