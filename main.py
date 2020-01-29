import sys
import tensorflow as tf


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
    model = tf.keras.models.load_model('model/fizz_buzz_model.h5')
    orig_stdout = sys.stdout
    f = open('Software2.txt', 'w')
    sys.stdout = f
    g = open(test_file, "r")
    test_inp = int(g)
    predictions = model.predict_classes(test_inp)
    for x in predictions:
        print(x)
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