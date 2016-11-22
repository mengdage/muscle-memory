import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
def main():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    X1, y1 = datasets.make_moons(200, noise=0.20)
    plt.figure(1)
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.figure(2)

    plt.scatter(X1[:,0], X1[:,1], c=y1, cmap=plt.cm.Spectral)
    plt.show()

if __name__ == "__main__":
    main()


    
