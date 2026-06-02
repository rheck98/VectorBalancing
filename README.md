# VectorBalancing
Code for running simulations for alpha-online vector balancing. There are three scripts performing three different functionalities. For each script, parameters can be modified at the beginning of the script and the code is set up to run the main function based on those parameters.

# boundary_iteration:

This script performs the discrete updates using the **toothpick** method with upper and lower corrections near the corners. The code plots the upper and lower approximations for each step, calculates the area change, and returns the mean and standard deviation of the area changes in the upper and lower approximations over the steps.

# diff_eq:

This script solves the differential equation for the shape which shrinks homothetically (with shrink rate C, modifiable in the code) under the curve-shortening flow. The code returns a plot of the shape.

# plot_shape_evolution:

This script plots the evolution of the eye-shape pixel-wise (using parallelization) over a fixed number of iterations and plots the results.

