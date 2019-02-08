import numpy as np


def one_hot_encode_int_array(array, num_cats):
    shape = tuple(list(array.shape) + [num_cats])
    one_hot = np.zeros(shape)
    for i in range(num_cats):
        one_hot[..., i] = np.where(array == i, 1, 0)
    return one_hot

'''
def one_hot_encode_int_array(array, num_cats):
    one_hot = None
    for i in range(num_cats):
        if one_hot is None:
            one_hot = np.where(array == i, 1, 0)
            one_hot = one_hot[..., np.newaxis]
        else:
            indices = np.where(array == i, 1, 0)
            indices = indices[..., np.newaxis]
            one_hot = np.concatenate((one_hot, indices), len(array.shape))
    return one_hot
'''

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

