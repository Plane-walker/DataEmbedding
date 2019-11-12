import matplotlib.pyplot as plt
import matplotlib.image as mp_img
import numpy as np
import math


def img_process(img, size):
    img_grey = img.min(axis=-1)
    x = math.ceil(img_grey.shape[0] / size) * size
    y = math.ceil(img_grey.shape[1] / size) * size
    out_img = np.zeros((x, y), dtype=np.int)
    out_img[0:img_grey.shape[0], 0:img_grey.shape[1]] = img_grey[0:img_grey.shape[0], 0:img_grey.shape[1]]
    return out_img


def img_cut(img, size):
    x = math.floor(img.shape[0] / size)
    y = math.floor(img.shape[1] / size)
    block_list = []
    for index_x in range(x):
        for index_y in range(y):
            block_list.append(img[index_x * size: (index_x + 1) * size, index_y * size: (index_y + 1) * size])
    return block_list, x, y


def get_block_size(img):
    return 64


def get_key(size):
    byte_sequence = (np.random.permutation(size)).tolist()
    return byte_sequence


def img_combine(block_list, size, x, y):
    img_x = size * x
    img_y = size * y
    out_img = np.zeros((img_x, img_y),dtype=np.int)
    for index_x in range(x):
        for index_y in range(y):
            out_img[index_x * size:(index_x + 1) * size, index_y * size:(index_y + 1) * size] = block_list[index_x * y + index_y]
    return out_img


def img_permutation(block_list, byte_sequence):
    result = []
    for index in range(len(byte_sequence)):
        result.append(block_list[byte_sequence.index(index)])
    return result


def img_recover(block_list, byte_sequence):
    result = []
    for index in range(len(byte_sequence)):
        result.append(block_list[byte_sequence[index]])
    return result


def stream_encryption(block_list, byte_sequence):
    out_key = stream_key(byte_sequence, len(byte_sequence) * len(block_list))
    for index in range(len(block_list)):
        x = block_list[index].shape[0]
        y = block_list[index].shape[1]
        sub_key = out_key[index * len(byte_sequence): (index + 1) * len(byte_sequence)]
        img_key = stream_key(sub_key, x * y)
        for index_x in range(x):
            for index_y in range(y):
                block_list[index][index_x, index_y] ^= img_key[index_x * y + index_y]
    return block_list


def stream_key(short_key, num):
    box = list(range(256))
    swap_index = 0
    for index in range(256):
        temp = box[index]
        swap_index = (swap_index + box[index] + short_key[index % len(short_key)]) % 256
        box[index] = box[swap_index]
        box[swap_index] = temp

    index_1 = index_2 = 0
    out_key = []
    for index in range(num):
        index_1 = (index_1 + 1) % 256
        index_2 = (index_2 + box[index_1]) % 256
        temp = box[index_1]
        box[index_1] = box[index_2]
        box[index_2] = temp
        out_key.append(box[(box[index_1] + box[index_2]) % 256])
    return out_key


def main():
    img_addr = input()
    origin_img = mp_img.imread(img_addr)
    block_size = get_block_size(origin_img)

    img = img_process(origin_img, block_size)
    plt.figure("Image")
    plt.imshow(img, 'Greys')
    plt.show()
    block_list, x_num, y_num = img_cut(img, block_size)

    key = get_key(len(block_list))
    encrypted_block_list = stream_encryption(img_permutation(block_list, key), key)
    encrypted_block_list = img_recover(stream_encryption(encrypted_block_list, key), key)
    img = img_combine(encrypted_block_list, block_size, x_num, y_num)

    plt.figure("Image")
    plt.imshow(img, 'Greys')
    plt.show()


if __name__ == '__main__':
    main()
