import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
from bitarray import bitarray


def list_key2string(key):
    return "".join(['{:02x}'.format(i) for i in key])


def string2list_key(key):
    tmp = []
    for index in range(len(key)):
        if index % 2 == 1:
            tmp.append(key[index - 1] + key[index])
    return [int(i, 16) for i in tmp]


def str2bit_array(s):
    ret = bitarray(''.join([bin(int('1' + hex(c)[2:], 16))[3:] for c in s.encode('utf-8')]))
    return ret


def bit_array2str(bit):
    return bit.tobytes().decode('utf-8')


def img_preprocess(origin_img):
    img = np.array(origin_img)
    size = 8
    x = math.ceil(img.shape[0] / size) * size
    y = math.ceil(img.shape[1] / size) * size
    print("total bits length: ", x * y)
    out_img = np.zeros((x, y), dtype=np.int)
    out_img[0:img.shape[0], 0:img.shape[1]] = img[0:img.shape[0], 0:img.shape[1]]
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
    byte_sequence = list((np.random.permutation(size)))
    return byte_sequence


def get_permutation_key(short_key, size):
    box = list(range(size))
    swap_index = 0
    for index in range(size):
        temp = box[index]
        swap_index = (swap_index + box[index] + short_key[index % len(short_key)]) % size
        box[index] = box[swap_index]
        box[swap_index] = temp
    return box


def img_combine(block_list, size, x, y):
    img_x = size * x
    img_y = size * y
    out_img = np.zeros((img_x, img_y), dtype=np.int)
    for index_x in range(x):
        for index_y in range(y):
            out_img[index_x * size:(index_x + 1) * size, index_y * size:(index_y + 1) * size] = block_list[index_x * y + index_y]
    return out_img


def img_permutation(block_list, key):
    result = []
    permutation_key = get_permutation_key(key, len(block_list))
    for index in range(len(permutation_key)):
        result.append(block_list[permutation_key.index(index)])
    return result


def img_recover(block_list, key):
    result = []
    permutation_key = get_permutation_key(key, len(block_list))
    for index in range(len(permutation_key)):
        result.append(block_list[permutation_key[index]])
    return result


def stream_encryption(block_list, key):
    img_key = stream_key(key, len(block_list))
    x = block_list[0].shape[0]
    y = block_list[0].shape[1]
    for index in range(len(block_list)):
        for index_x in range(x):
            for index_y in range(y):
                block_list[index][index_x, index_y] ^= img_key[index]


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


def histogram_shifting(block_list, embed_bits, key):
    out_key = stream_key(key, len(key) * len(block_list))
    h = []
    for index in range(len(block_list)):
        h.extend(pixel_position(block_list[index]))
    embed_bits.append(1)
    print("length of Map: ", len(h))
    h.extend(embed_bits)
    x = block_list[0].shape[0]
    y = block_list[0].shape[1]
    for index in range(len(block_list)):
        sub_key = out_key[index * len(key): (index + 1) * len(key)]
        img_key = get_permutation_key(sub_key, x * y)
        data_embed(block_list[index], img_key, h)
    if len(h) == 0:
        return True
    return False


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


def data_recover(block_list, key):
    out_key = stream_key(key, len(key) * len(block_list))
    h = []
    x = block_list[0].shape[0]
    y = block_list[0].shape[1]
    for index in range(len(block_list)):
        sub_key = out_key[index * len(key): (index + 1) * len(key)]
        img_key = get_permutation_key(sub_key, x * y)
        h.extend(bits_recover(block_list[index], img_key))
    for index in range(len(block_list)):
        pixel_position_recover(block_list[index], h)
    print("max hidden bits length: ", len(h))
    while h.pop() != 1:
        pass
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
    plt.figure("Image")
    plt.imshow(origin_img)
    plt.show()
    rgb_img = origin_img.split()

    img_r, block_size = img_preprocess(rgb_img[0])
    img_g = img_preprocess(rgb_img[1])[0]
    img_b = img_preprocess(rgb_img[2])[0]
    block_list_r, x_num, y_num = img_cut(img_r, block_size)
    block_list_g = img_cut(img_g, block_size)[0]
    block_list_b = img_cut(img_b, block_size)[0]

    key = get_key(128)
    block_list_r = img_permutation(block_list_r, key)
    stream_encryption(block_list_r, key)
    block_list_g = img_permutation(block_list_g, key)
    stream_encryption(block_list_g, key)
    block_list_b = img_permutation(block_list_b, key)
    stream_encryption(block_list_b, key)
    embed_key = get_key(128)
    embed_bits = str2bit_array("plane-walker").tolist()
    if not histogram_shifting(block_list_r, embed_bits, embed_key):
        print("space run out.")
        return

    img_r = Image.fromarray(np.uint8(img_combine(block_list_r, block_size, x_num, y_num)))
    img_g = Image.fromarray(np.uint8(img_combine(block_list_g, block_size, x_num, y_num)))
    img_b = Image.fromarray(np.uint8(img_combine(block_list_b, block_size, x_num, y_num)))
    rgb_img = Image.merge('RGB', (img_r, img_g, img_b))
    plt.figure("Image")
    plt.imshow(rgb_img)
    plt.show()
    string_key = list_key2string(key)
    string_embed_key = list_key2string(embed_key)
    print("encryption key: " + string_key)
    print("embed key: " + string_embed_key)

    rgb_img = rgb_img.split()
    img_r, block_size = img_preprocess(rgb_img[0])
    img_g = img_preprocess(rgb_img[1])[0]
    img_b = img_preprocess(rgb_img[2])[0]
    block_list_r, x_num, y_num = img_cut(img_r, block_size)
    block_list_g = img_cut(img_g, block_size)[0]
    block_list_b = img_cut(img_b, block_size)[0]
    key = string2list_key(string_key)
    embed_key = string2list_key(string_embed_key)

    recover_data = data_recover(block_list_r, embed_key)
    print(bit_array2str(bitarray(recover_data)))

    stream_encryption(block_list_r, key)
    stream_encryption(block_list_g, key)
    stream_encryption(block_list_b, key)

    block_list_r = img_recover(block_list_r, key)
    block_list_g = img_recover(block_list_g, key)
    block_list_b = img_recover(block_list_b, key)

    img_r = Image.fromarray(np.uint8(img_combine(block_list_r, block_size, x_num, y_num)))
    img_g = Image.fromarray(np.uint8(img_combine(block_list_g, block_size, x_num, y_num)))
    img_b = Image.fromarray(np.uint8(img_combine(block_list_b, block_size, x_num, y_num)))
    final_img = Image.merge('RGB', (img_r, img_g, img_b))
    plt.figure("Image")
    plt.imshow(final_img)
    plt.show()


if __name__ == '__main__':
    main()
