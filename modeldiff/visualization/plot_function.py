from numpy import exp,arange
import numpy as np
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

from matplotlib import pyplot as plt

import pprint

# the function that I'm going to plot
def z_func1(x,y):
    return -((10-x)**2 + (10-y)**2)

def z_func2(x,y):
    return -((15-x)**2 + (15-y)**2)

# def z_func2(x,y):
#     return -((10-x)**2 + (10-y)**2) + 0.5*((2-x)**2 + (3-y)**2)

def value_p1(x,y,gx,gy,cx,cy,t=20):

    if x > gx or y > gy:
        return None

    total_r = 0

    x_list = [(i, y) for i in range(x, gx+1)]
    y_list = [(gx, i) for i in range(y+1, gy+1)]


    total_list = list(set(x_list + y_list))

    do_nothing_times = t - len(total_list)

    for i in total_list:
        total_r += -(abs(cx - i[0]) + abs(cy - i[1]))

    total_r += do_nothing_times * -(abs(cx - gx) + abs(cy - gy))
    return total_r





def z_func3(x,y):
    r_matrix = -((10-x)**2 + (10-y)**2)
    r_matrix[2:5,2:5] = -500
    return r_matrix
 
x = arange(0,30,1)
y = arange(0,30,1)

# value_p1(0,0,9,9,3,3)
# value_p1(0,3,9,9,3,3)
# raise ValueError


Z_0 = np.zeros((10,10))
for i in range(0, Z_0.shape[0]):
    for j in range(0, Z_0.shape[1]):
        Z_0[j][i] = value_p1(i,j,9,9,5,5)

Z_1 = np.zeros((10,10))
for i in range(0, Z_1.shape[0]):
    for j in range(0, Z_1.shape[1]):
        Z_1[j][i] = value_p1(i,j,9,9,9,9)

Z_2 = abs(Z_0 - Z_1)

fig, ax = plt.subplots(figsize=(7,7))

X,Y = meshgrid(x, y) # grid of point
Z = z_func1(X, Y) # evaluation of the function on the grid
Z2 = z_func2(X, Y)
Z3 = Z2 - Z
Z4 = z_func3(X,Y)

plt.xticks(x)
plt.yticks(y)

im = plt.imshow(Z_1,cmap=cm.binary) # drawing the function
# adding the Contour lines with labels
# cset = contour(Z,arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
# clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
plt.colorbar(im) # adding the colobar on the right
# # latex fashion title
title('Value Map')
plt.show()