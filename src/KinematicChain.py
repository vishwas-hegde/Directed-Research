import ompl.base as ob
import ompl.geometric as og
from ompl import util as ou
from ompl import tools as ot
import numpy as np

import KinematicHeader
from KinematicHeader import KinematicChainValidityChecker
from KinematicHeader import KinematicChainSpace
from Maps import createTestEnvironment
from Maps import ReadEnvironmentFromFile
import os
import omplVisualize
from time import time
import math
from nevergrad.optimization import optimizerlib
import nevergrad as ng

Map_Num = 0

def optimize_param(Range, Goal_Bias, Proj_Links):
    '''
    # Info:
    # si: Space Infomation
    # env: Environment to be used. Contains the map to be used. Contains segments representing boundary and obstacles.
    # numlinks: The number of links present in robot
    # ValidityChecker: Collision Check is implemented here. This checks if the randomly sampled state is valid.
    '''

    numLinks = int(8)   # change number of links here
    benchmarking = 'OFF'
    current_planner = 'KPIECE'
    
    
    print("Range:",Range)
    print("Goal Bias:",Goal_Bias)
    print("Number of Links for projection:",Proj_Links)

    env = ReadEnvironmentFromFile(Map_Num)
    # change link-length here
    chain = KinematicChainSpace(num_links=numLinks, link_length=3.0 / float(numLinks), env=env, proj_link_count = Proj_Links)

    # Use simple setup. Can be done without simpleSetup as well. Check file RigidBodyPlanning.py for example.
    ss = og.SimpleSetup(chain)

    # Get Space Validity checker defined in KinematicHeader.py.
    validity_checker = KinematicChainValidityChecker(ss.getSpaceInformation())

    # Set space validity Checker. This is where all collision checking will be done.
    ss.setStateValidityChecker(validity_checker)

    start = ob.State(chain)
    goal = ob.State(chain)
    
    chain.setup()    
    
    for i in range(numLinks):
        start[i] = 0.
        goal[i] = 0.
        
    goal[0] = math.pi    
    
    # Set the start and goal states in the state space.
    ss.setStartAndGoalStates(start, goal)

    si = ss.getSpaceInformation()
    
    if current_planner == 'RRT':
        planner = og.RRTConnect(si)
        planner.setRange(2.8)   # Set the range of RRTConnect
    
    elif current_planner == 'KPIECE':
        planner = og.KPIECE1(si)    # Change Planner Here
        planner.setRange(Range)   # Set the range of KPIECE
        planner.setGoalBias(Goal_Bias)   # Set the goal bias of KPIECE
    
    
    ss.setPlanner(planner)
    times = []
    for i in range(10):
        start_time = time() 
        solved = ss.solve(10.0)
        end_time = time()
        times.append(float(abs(start_time - end_time)))
    mean_time = np.mean(times)
    return mean_time

def objective_function(Range, Goal_Bias, Num_Links):
    
    return optimize_param(Range, Goal_Bias, Num_Links)

param = {}
Num_Maps_To_Optimize = 2

for i in range(Num_Maps_To_Optimize):
    
    # Define the parameter space
    param1 = ng.p.Scalar(lower=0.01, upper=10)      # Range
    param2 = ng.p.Scalar(lower=0.01, upper=0.7)     # Goal Bias
    param3 = ng.p.Choice([1, 2, 3, 4, 5, 6, 7, 8])  # Number of Links for projection

    # Combine parameters into a list
    param_list = [param1, param2, param3]

    # Set the instrumentation
    instrumentation = ng.p.Instrumentation(*param_list)
    
    # Define the optimization problem
    optimizer = ng.optimizers.OnePlusOne(parametrization=instrumentation, budget=100)
    
    # Run the optimization process
    recommendation = optimizer.minimize(objective_function)

    # Retrieve the optimal value of x
    optimal_x = recommendation.value

    param[i] = optimal_x
    Map_Num += 1

print(param)
with open('param.txt', 'w') as file:
    file.write(str(param))
