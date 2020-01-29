import sys


def main():
    START = 1
    END = 10
    orig_stdout = sys.stdout
    f = open('train_1.txt', 'w')
    sys.stdout = f
    for i in range(START, END + 1):
        print(i)
    sys.stdout = orig_stdout
    f.close()


if __name__ == "__main__":
    main()