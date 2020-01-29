import matplotlib.pyplot as plt

x = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16]
y = [100, 98, 85, 91, 93, 61, 65, 94, 73, 97, 67]

plt.plot(x, y)
plt.xlabel("BASE")
plt.ylabel("Accuracy (in %)")
plt.show()