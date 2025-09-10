#import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interactive

#Function to update and display the plot

def update_plot(hidden_layer_size):
    #Generate synthetic data (circle)
    X, y = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=0)

    #Create a multi-layer perceptron (MLP) classifier
    clf = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,),activation='relu', max_iter=3000, random_state=1)

    #Fit the classifier to data
    clf.fit(X, y)

    #create a grid of points for visualization
    x_vals = np.linspace(X[:, 0].min() -0.1, X[:, 0].max() + 0.1, 100)

    y_vals = np.linspace(X[:, 1].min() - 0.1, X[:,1].max() + 0.1, 100)

    X_plane, y_plane = np.meshgrid(x_vals, y_vals)

    grid_points = np.column_stack((X_plane.ravel(), y_plane.ravel()))

    #Predict class labels for the grid points
    Z = clf.predict(grid_points)
    Z = Z.reshape(X_plane.shape)
    #Clear previous plot
    plt.clf()

    #Plot the decision boundary
    plt.contourf(X_plane, y_plane, Z, levels=[-0.5, 0.5, 1.5],cmap=plt.cm.RdYlGn, alpha=0.6)
    plt.scatter(X[:, 0], X[:,1], c=y, cmap=plt.cm.RdYlGn)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Decision Boundary Visualization for Circle Data(Hidden Layer Size={hidden_layer_size})')
    plt.show()
    
# #Create a slider for hidden layer size
hidden_layer_size_slider = widgets.IntSlider(value=2, min=1, max=10, step=1, description='Hidden Layer Size')

# Creaete a submit button
submit_button = widgets.Button(description='Submit')

#Define a callback function for the submit button

def submit_button_callback(button):
    update_plot(hidden_layer_size_slider.value)

#submit_button.on_click(submit_button_callback)

#Create an interactive widget
interactive_plot = interactive(update_plot,hidden_layer_size=hidden_layer_size_slider)

#Display the widgets
display(interactive_plot)



