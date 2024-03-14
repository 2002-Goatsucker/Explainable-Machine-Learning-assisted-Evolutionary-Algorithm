from dependency import *
from mpl_toolkits.mplot3d import Axes3D

def paint3D(x,y,z):
    fig = plt.figure(figsize = (10, 7))  
    ax = plt.axes(projection ="3d")
    ax.scatter3D(x, y, z, color = "blue", s=1)
    plt.title("3D scatter plot")  
    plt.show()