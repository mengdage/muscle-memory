"""
a simple classification according to the one written by
http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
"""
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

def generate_data():
    """Generate datasets
    Args:
        NONE
    Returns:
        X and y
    """
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def plot_decision_boundary(pred_func, X, y):
    """Plot decision boundary
    Args:
        pred_func
        X
        y
    Returns:
        NONE
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.1 # step size in the mesh
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the countour and training examples
    plt.figure(1)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

def visualize(X, y, clf):
    """Visualize 

    """
    #plot_decision_boundary(lambda x: clf.predict(x), X, y)
    plot_decision_boundary(lambda x: clf.predict(x), X, y)


    plt.title("Logistic Regression")

def classify(X, y):
    """Classify
    """
    clf = linear_model.LogisticRegressionCV()
    clf.fit(X, y)
    return clf

def main():
    """Main
    """
    X, y = generate_data()
    clf = classify(X, y)
    visualize(X, y, clf)

if __name__ == "__main__":
    main()
