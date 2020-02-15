import sys
import tensorflow as tf
import sys
import numpy as np
BASE = 2
MAX_VAL = 1000

def base_len(size, base):
    i = 0
    while size != 0:
        i = i+1
        size = int(size / base)
    return i
    
def get_num(inp_arr):
    x = 0
    for i in range(len(inp_arr) - 1, 0, -1):
        x = x + ((BASE**i) * inp_arr[i])
    return x


def structurize_data(data, size):
    str_data = np.ndarray((len(data), size), dtype='float')
    for i in range(len(data)):
        for j in range(size):
            x = data[i]%(BASE ** (j+1))
            str_data[i][j] = (int(x/(BASE ** j)))
    return str_data


def get_label(label, val):
    if label == 0:
        return "fizzbuzz"
    if label == 1:
        return "fizz"
    if label == 2:
        return "buzz"
    if label == 3:
        return str(val)


def fizz_buzz_logic(val):
    if val % 15 == 0:
        print("fizzbuzz")
    elif val % 3 == 0:
        print("fizz")
    elif val % 5 == 0:
        print("buzz")
    else:
        print(val)


def soft_1(test_file):
    orig_stdout = sys.stdout
    f = open('Software1.txt', 'w')
    sys.stdout = f
    g = open(test_file, "r")
    for x in g:
        fizz_buzz_logic(int(x))
    sys.stdout = orig_stdout
    g.close()
    f.close()


def soft_2(test_file):
    size = base_len(MAX_VAL, BASE) + 1
    //test_file = "test_input.txt"
    model = tf.keras.models.load_model('model/fizz_buzz_model.h5')
    orig_stdout = sys.stdout
    f = open('Software2.txt', 'w')
    sys.stdout = f
    g = open(test_file, "r")
    test_inp_num = [int(i) for i in g]
    test_inp_arr = structurize_data(test_inp_num, size)
    predictions = model.predict_classes(test_inp_arr)
    i = 0
    for x in predictions:
        print(get_label(x, test_inp_num[i]))
        i = i + 1
    sys.stdout = orig_stdout
    g.close()
    f.close()


def main():
    i = 0
    test_file_name = "test_input.txt"
    for args in sys.argv:
        i = i+1
        if args == "--test-data":
            test_file_name = sys.argv[i]
            break
    soft_1(test_file_name)
    soft_2(test_file_name)



if __name__ == "__main__":
    main()