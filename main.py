import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
from bitarray import bitarray

MIN_SIZE = 64


def str2bit_array(s):
    ret = bitarray(''.join([bin(int('1' + hex(c)[2:], 16))[3:] for c in s.encode('utf-8')]))
    return ret


def bit_array2str(bit):
    return bit.tobytes().decode('utf-8')


def img_preprocess(img):
    width, height = img.size

    size = MIN_SIZE * round(width * height / 200 / (MIN_SIZE ** 2))
    if size == 0:
        size = MIN_SIZE
    # img_grey = np.array(img.convert('L'))
    rgb_img = img.split()
    img_grey = np.array(rgb_img[0])
    x = math.ceil(img_grey.shape[0] / size) * size
    y = math.ceil(img_grey.shape[1] / size) * size
    out_img = np.zeros((x, y), dtype=np.int)
    out_img[0:img_grey.shape[0], 0:img_grey.shape[1]] = img_grey[0:img_grey.shape[0], 0:img_grey.shape[1]]
    return out_img, size


def img_cut(img, size):
    x = int(img.shape[0] / size)
    y = int(img.shape[1] / size)
    block_list = []
    for index_x in range(x):
        for index_y in range(y):
            block_list.append(img[index_x * size: (index_x + 1) * size, index_y * size: (index_y + 1) * size])
    return block_list, x, y


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
    img_key = stream_key(byte_sequence, len(block_list), 256)
    for index in range(len(block_list)):
        x = block_list[index].shape[0]
        y = block_list[index].shape[1]
        for index_x in range(x):
            for index_y in range(y):
                block_list[index][index_x, index_y] ^= img_key[index]
    return block_list


def stream_key(short_key, num, range_size):
    box = list(range(range_size))
    swap_index = 0
    for index in range(range_size):
        temp = box[index]
        swap_index = (swap_index + box[index] + short_key[index % len(short_key)]) % range_size
        box[index] = box[swap_index]
        box[swap_index] = temp

    index_1 = index_2 = 0
    out_key = []
    for index in range(num):
        index_1 = (index_1 + 1) % range_size
        index_2 = (index_2 + box[index_1]) % range_size
        temp = box[index_1]
        box[index_1] = box[index_2]
        box[index_2] = temp
        out_key.append(box[(box[index_1] + box[index_2]) % range_size])
    return out_key


def histogram_shifting(block_list, embed_bits, byte_sequence):
    out_key = stream_key(byte_sequence, len(byte_sequence) * len(block_list), 256)
    h = []
    for index in range(len(block_list)):
        h.extend(pixel_position(block_list[index]))
    h.extend(embed_bits)
    print(len(h))
    for index in range(len(block_list)):
        x = block_list[index].shape[0]
        y = block_list[index].shape[1]
        sub_key = out_key[index * len(byte_sequence): (index + 1) * len(byte_sequence)]
        img_key = stream_key(sub_key, x * y, x * y)
        data_embed(block_list[index], img_key, h)
    print(len(h))


def pixel_position(block):
    x = block.shape[0]
    y = block.shape[1]
    h = []
    for index_x in range(x):
        for index_y in range(y):
            if block[index_x, index_y] == 1 or block[index_x, index_y] == 254:
                h.append(1)
            elif block[index_x, index_y] == 0:
                h.append(0)
                block[index_x, index_y] = 1
            elif block[index_x, index_y] == 255:
                h.append(0)
                block[index_x, index_y] = 254
    return h


def data_embed(block, key, hidden_data):
    x = block.shape[0]
    y = block.shape[1]
    first_peak_x = math.floor(key[0] / x)
    first_peak_y = key[0] - first_peak_x * x
    second_peak_x = math.floor(key[1] / x)
    second_peak_y = key[1] - second_peak_x * x
    first = block[first_peak_x, first_peak_y]
    second = block[second_peak_x, second_peak_y]
    smaller = first if first <= second else second
    bigger = first if first > second else second
    for index_x in range(x):
        for index_y in range(y):
            if index_x != first_peak_x and index_x != second_peak_x and index_y != first_peak_y and index_y != second_peak_y:
                if block[first_peak_x, first_peak_y] == block[second_peak_x, second_peak_y]:
                    if block[index_x, index_y] < smaller:
                        block[index_x, index_y] -= 1
                    elif block[index_x, index_y] == smaller:
                        if len(hidden_data) > 0:
                            block[index_x, index_y] -= hidden_data.pop(0)
                else:
                    if block[index_x, index_y] < smaller:
                        block[index_x, index_y] -= 1
                    elif block[index_x, index_y] == smaller:
                        if len(hidden_data) > 0:
                            block[index_x, index_y] -= hidden_data.pop(0)
                    elif block[index_x, index_y] == bigger:
                        if len(hidden_data) > 0:
                            block[index_x, index_y] += hidden_data.pop(0)
                    elif block[index_x, index_y] > bigger:
                        block[index_x, index_y] += 1


def data_recover(block_list, byte_sequence):
    out_key = stream_key(byte_sequence, len(byte_sequence) * len(block_list), 256)
    h = []
    for index in range(len(block_list)):
        x = block_list[index].shape[0]
        y = block_list[index].shape[1]
        sub_key = out_key[index * len(byte_sequence): (index + 1) * len(byte_sequence)]
        img_key = stream_key(sub_key, x * y, x * y)
        h.extend(bits_recover(block_list[index], img_key))
    for index in range(len(block_list)):
        pixel_position_recover(block_list[index], h)
    return h


def bits_recover(block, key):
    hidden_data = []
    x = block.shape[0]
    y = block.shape[1]
    first_peak_x = math.floor(key[0] / x)
    first_peak_y = key[0] - first_peak_x * x
    second_peak_x = math.floor(key[1] / x)
    second_peak_y = key[1] - second_peak_x * x
    first = block[first_peak_x, first_peak_y]
    second = block[second_peak_x, second_peak_y]
    smaller = first if first <= second else second
    bigger = first if first > second else second
    for index_x in range(x):
        for index_y in range(y):
            if index_x != first_peak_x and index_x != second_peak_x and index_y != first_peak_y and index_y != second_peak_y:
                if block[first_peak_x, first_peak_y] == block[second_peak_x, second_peak_y]:
                    if block[index_x, index_y] < smaller:
                        if block[index_x, index_y] == smaller - 1:
                            hidden_data.append(1)
                        block[index_x, index_y] += 1
                    elif block[index_x, index_y] == smaller:
                        hidden_data.append(0)
                else:
                    if block[index_x, index_y] < smaller:
                        if block[index_x, index_y] == smaller - 1:
                            hidden_data.append(1)
                        block[index_x, index_y] += 1
                    elif block[index_x, index_y] == smaller or block[index_x, index_y] == bigger:
                        hidden_data.append(0)
                    elif block[index_x, index_y] > bigger:
                        if block[index_x, index_y] == bigger + 1:
                            hidden_data.append(1)
                        block[index_x, index_y] -= 1
    return hidden_data


def pixel_position_recover(block, h):
    x = block.shape[0]
    y = block.shape[1]
    for index_x in range(x):
        for index_y in range(y):
            if block[index_x, index_y] == 1:
                if (h.pop(0)) == 0:
                    block[index_x, index_y] = 0
            elif block[index_x, index_y] == 254:
                if (h.pop(0)) == 0:
                    block[index_x, index_y] = 255


def main():
    img_addr = input()
    origin_img = Image.open(img_addr)
    img, block_size = img_preprocess(origin_img)
    plt.figure("Image")
    plt.imshow(img, 'Greys')
    plt.show()
    block_list, x_num, y_num = img_cut(img, block_size)

    key = get_key(len(block_list))
    encrypted_block_list = stream_encryption(img_permutation(block_list, key), key)
    embed_key = get_key(len(block_list))
    embed_bits = str2bit_array("plane-walker").tolist()
    embed_bits.append(1)
    histogram_shifting(block_list, embed_bits, embed_key)
    recover_data = data_recover(block_list, embed_key)
    while recover_data.pop() != 1:
        pass
    print(bit_array2str(bitarray(recover_data)))
    encrypted_block_list = img_recover(stream_encryption(encrypted_block_list, key), key)
    img = img_combine(encrypted_block_list, block_size, x_num, y_num)

    plt.figure("Image")
    plt.imshow(img, 'Greys')
    plt.show()


if __name__ == '__main__':
    main()
