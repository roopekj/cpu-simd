from matplotlib import pyplot as plt

with open("data.txt", "r") as f:
    data = [float(line.strip()) for line in f.readlines()]

cpu = [a for a in data[::2]]
gpu = [a for a in data[1::2]]
xrange = [a * 2 for a in range(int(len(cpu)))]

plt.plot(xrange, cpu)
plt.ylim(0, 0.005)
plt.plot(xrange, gpu)
plt.show()
