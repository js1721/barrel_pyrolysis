import csv
import matplotlib.pyplot as plt

x = []
phi = []

with open("solution.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:  # skip empty rows
            continue
        x.append(float(row[0]))
        phi.append(float(row[1]))

plt.plot(x, phi, marker="o")
plt.xlabel("x")
plt.ylabel("Flux φ(x)")
plt.title("Neutron Flux Distribution")
plt.grid(True)
plt.show()
