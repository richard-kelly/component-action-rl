import numpy as np


def one_hot_encode_int_array(array, num_cats):
    shape = tuple([num_cats] + list(array.shape))
    one_hot = np.zeros(shape)
    for i in range(num_cats):
        indices = np.argwhere(array == i)
        for j in indices:
            one_hot[tuple([i]+j.tolist())] = 1.0
    return one_hot


def main():
    print("testing one hot encode:")
    x = np.arange(0, 9, 1).reshape(3, 3)
    print('x:')
    print(x)
    print('one_hot_encode_int_array(x, 9)')
    print(one_hot_encode_int_array(x, 9))


if __name__ == "__main__":
    main()

