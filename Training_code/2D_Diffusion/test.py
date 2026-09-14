import numpy as np 
import matplotlib.pyplot as plt


# def S(x): 
#     return 2*np.pi*np.pi*np.cos(np.pi *x)*np.cos(np.pi *x)



# M = np.loadtxt("build/diff_matrix.csv", delimiter=",")
# rhs = np.loadtxt("build/rhs.csv", delimiter=",")




data = np.loadtxt("build/flux.txt")

x = data[:, 0]
y = data[:, 1]
f = data[:, 2]

plt.contour(x,y,f)