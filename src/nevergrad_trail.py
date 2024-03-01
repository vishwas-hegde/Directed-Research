import KinematicChain
import nevergrad as ng
from nevergrad.optimization import optimizerlib

def objective_function(x):
    return KinematicChain.optimze_param(x)

# Define the optimization domain for parameter x
# Here we set a range for x from 0.01 to 10
# Adjust the range as per your requirements
# param_space = ng.p.Scalar(lower=0.01, upper=10)
param = {}
for i in range(10):
    param_space = ng.p.Array(shape=(2,), lower=[0.01, 0.01], upper=[10, 0.8])

    # Choose the optimizer
    optimizer = optimizerlib.OnePlusOne(parametrization=param_space, budget=30)

    # Run the optimization process
    recommendation = optimizer.minimize(objective_function)

    # Retrieve the optimal value of x
    optimal_x = recommendation.value

    # Print the optimal value of x and corresponding execution time
    print("Optimal value of x:", optimal_x)
    # print("Corresponding execution time:", objective_function(optimal_x))