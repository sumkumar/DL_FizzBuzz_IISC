import numpy as np
import tensorflow as tf

TRAIN_START = 101
TRAIN_END = 1000
CLASS_SIZE = 4
BASE = 3
NUM_EPOCHS = 1000


def fizz_buzz_out(train_inp):
    train_out = np.ndarray((len(train_inp), 4))
    ind = 0
    for i in train_inp:
        if i % 15 == 0:
            out = [1, 0, 0, 0]
        elif i % 3 == 0:
            out = [0, 1, 0, 0]
        elif i % 5 == 0:
            out = [0, 0, 1, 0]
        else:
            out = [0, 0, 0, 1]
        train_out[ind] = out
        ind += 1
    return train_out


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)


def get_randomized_weights():
    w_h = init_weights([900, NUM_HIDDEN])
    w_o = init_weights([NUM_HIDDEN, 1])
    return w_h, w_o


def train_model(py_x, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(py_x, Y))
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
    return train_op


def normalize_data(train_data, test_data):
    train_data, test_data = train_data / (CLASS_SIZE - 1), test_data / (CLASS_SIZE - 1)
    return train_data, test_data


def structurize_data(data, size):
    str_data = np.ndarray((len(data), size), dtype='float')
    for i in range(len(data)):
        for j in range(size):
            x = data[i] % (BASE ** (j + 1))
            str_data[i][j] = (int(x / (BASE ** j)))
    return str_data


def base_len(size, base):
    i = 0
    while size != 0:
        i = i + 1
        size = int(size / base)
    return i


def get_label(label, val):
    if label == 0:
        return "FizzBuzz"
    if label == 1:
        return "Fizz"
    if label == 2:
        return "Buzz"
    if label == 3:
        return str(val)


def fizz_buzz_NN():
    size = base_len(TRAIN_END, BASE) + 1
    inp = np.array(range(TRAIN_START, TRAIN_END + 1), dtype='float')
    tst = np.array(range(1, TRAIN_START), dtype='float')
    train_inp = structurize_data(inp, size)  # inp
    train_out = fizz_buzz_out(inp)
    test_inp = structurize_data(tst, size)  # tst
    test_out = fizz_buzz_out(tst)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(CLASS_SIZE, activation='softmax')
    ])

    model.compile(optimizer='RMSprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # print(model.get_weights())
    model.fit(train_inp, train_out, epochs=NUM_EPOCHS)
    model.save("model/fizz_buzz_model.h5")
    model.evaluate(test_inp, test_out, verbose=2)
    predictions = model.predict_classes(test_inp)
    print(predictions[0])
    for i in range(1, TRAIN_START):
        print(str(i) + " " + get_label(predictions[i - 1], i))

