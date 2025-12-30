from matplotlib import pyplot as plt

with open("data.txt", "r") as f:
    data = [float(line.strip()) for line in f.readlines()]

cpu = [a for a in data[::2]]
gpu = [a for a in data[1::2]]
xrange = [(a + 1) * 2 for a in range(int(len(cpu)))]

font = {"size": 24}

plt.rc("font", **font)

plt.plot(xrange, cpu)
plt.xlim(1, 1024)
plt.ylim(0, 0.002)
plt.plot(xrange, gpu)
plt.show()
