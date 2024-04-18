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
from logger import SQLiteLogger
import multiprocessing
from pyswarms.single.global_best import GlobalBestPSO as PSO
import json


Map_Num = 0
# parameter_file_path = r"./parameters/map_"
Logger = SQLiteLogger("Logger500.db")
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
    
    Proj_Links = int(Proj_Links)
    if Proj_Links == 0:
        Proj_Links = 1
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
        planner.setRange(Range)   # Set the range of RRTConnect
    
    elif current_planner == 'KPIECE':
        planner = og.KPIECE1(si)    # Change Planner Here
        planner.setRange(Range)   # Set the range of KPIECE
        planner.setGoalBias(Goal_Bias)   # Set the goal bias of KPIECE
    
    
    ss.setPlanner(planner)
    times = []
    # file = parameter_file_path + str(Map_Num) + ".txt"
    # f = open(file, "w")
    Counter = 0
    for i in range(10):
        Break_Flag = False
        while(Break_Flag == False):
            start_time = time() 
            solved = ss.solve(5.0)
            end_time = time()
            Status = ss.haveExactSolutionPath()
            if Status == True or Counter == 5:
                Break_Flag = True
            else:
                Counter += 1
            times.append(float(abs(start_time - end_time)))
            planner.clear()
            params = [Range, Goal_Bias, Proj_Links]
            Logger.log(params, Status, Map_Num, float(abs(start_time - end_time)))
            # f.write(str(params) + " ," + str(Status) + " ," + str(Map_Num) + str(float(abs(start_time - end_time))) + "\n")
    mean_time = np.mean(times)
    # f.close()
    return mean_time

def objective_function(x):
    Results = []
    for i in range(10):
        Range, Goal_Bias, Proj_Links = x[i]
        Result = optimize_param(Range, Goal_Bias, Proj_Links)
        Results.append(Result)

    return Results

def multiprocess():
    """
    Put your optimization code here
    """
    range_constraint = (0.01, 10.0)
    bias_constraint = (0.0, 1.0)
    projection_constraint = (1, 8)
    
    # Define bounds
    lb = [range_constraint[0], bias_constraint[0], projection_constraint[0]]  # Lower bounds
    ub = [range_constraint[1], bias_constraint[1], projection_constraint[1]]  # Upper bounds

    bounds = (lb, ub)

    # Set up the optimizer
    options = {'c1': 0.5, 'c2': 0.5 , 'w':0.8}
    optimizer = PSO(n_particles=10, dimensions=3, options=options, bounds=bounds)
    Final_Dict = {}
    
    # Perform optimization
    best_cost, best_pos = optimizer.optimize(objective_function, iters=20)
    Opt_Range, Opt_Goal_Bias, Opt_Proj_Links = best_pos
    print('Map:', Map_Num)
    print('Range:', Opt_Range)
    print('Goal Bias:', Opt_Goal_Bias)
    print('Projection Links:', Opt_Proj_Links)
    Dict = {}
    Dict['Range'] = Opt_Range
    Dict['Bias'] = Opt_Goal_Bias
    Dict['Proj'] = Opt_Proj_Links
    Dict['Best_Cost'] = best_cost
    Final_Dict[str(i)] = Dict
    f = open('/home/dhrumil/Git/Directed-Research/src/PSO_Output/PSO_500_'+str(Map_Num)+'.json','w')
    json.dump(Final_Dict, indent=4, fp=f)
    f.close()

if __name__ == "__main__":

    processes = []
    for i in range(220,235):
        Map_Num = i
        p = multiprocessing.Process(target=multiprocess, args=())
        processes.append(p)
        p.start()
        
    for process in processes:
        process.join()
    