import numpy as np
import tensorflow as tf


TRAIN_START = 101
TRAIN_END = 1000
NUM_HIDDEN = 100


def fizz_buzz_logic(n):
    for i in range(1, n+1):
        if i % 15 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)


def fizz_buzz_out(train_inp):
    train_out = train_inp.copy()
    ind = 0
    for i in train_inp:
        if i % 15 == 0:
            out = -3
        elif i % 3 == 0:
            out = -1
        elif i % 5 == 0:
            out = -2
        else:
            out = i
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
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
    return train_op


def fizz_buzz_NN():
    train_inp = np.array(range(101, 121))
    train_out = fizz_buzz_out(train_inp)
    X = tf.placeholder("float", [None, 900])
    Y = tf.placeholder("float", [None, 1])
    w_h, w_o = get_randomized_weights()
    py_x = model(X, w_h, w_o)
    train_op = train_model(py_x, Y)
    predict_op = tf.argmax(py_x, 1)
    BATCH_SIZE = 128
    trX = train_inp
    trY = train_out
    # Launch the graph in a session
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        for epoch in range(10000):
            # Shuffle the data before each training iteration.
            p = np.random.permutation(range(len(trX)))
            trX, trY = trX[p], trY[p]

            # Train in batches of 128 inputs.
            for start in range(0, len(trX), BATCH_SIZE):
                end = start + BATCH_SIZE
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

            # And print the current accuracy on the training data.
            print(epoch, np.mean(np.argmax(trY, axis=1) ==
                                 sess.run(predict_op, feed_dict={X: trX, Y: trY})))

        # And now for some fizz buzz
        numbers = np.arange(1, 101)
        teX = np.transpose(binary_encode(numbers, 900))
        teY = sess.run(predict_op, feed_dict={X: teX})
        output = np.vectorize(fizz_buzz)(numbers, teY)

        print(output)


def main():
    fizz_buzz_NN()

if __name__ == "__main__":
    main()