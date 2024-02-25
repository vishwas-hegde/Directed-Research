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

if __name__ == "__main__":
    import math
    '''
    # Info:
    # si: Space Infomation
    # env: Environment to be used. Contains the map to be used. Contains segments representing boundary and obstacles.
    # numlinks: The number of links present in robot
    # ValidityChecker: Collision Check is implemented here. This checks if the randomly sampled state is valid.
    '''

    numLinks = int(8)   # change number of links here
    benchmarking = 'ON'
    current_planner = 'KPIECE'

    for Map_Num in range(50):
        print("Link_Length:",3.0/float(numLinks))
        # env = createHornEnvironment(d=numLinks, eps=math.log(float(numLinks)) / float(numLinks))
        # env = createTestEnvironment(d=numLinks)
        env = ReadEnvironmentFromFile(Map_Num)
        # change link-length here
        chain = KinematicChainSpace(num_links=numLinks, link_length=3.0 / float(numLinks), env=env, proj_chain_count = 3)

        # Use simple setup. Can be done without simpleSetup as well. Check file RigidBodyPlanning.py for example.
        ss = og.SimpleSetup(chain)

        # Get Space Validity checker defined in KinematicHeader.py.
        validity_checker = KinematicChainValidityChecker(ss.getSpaceInformation())

        # Set space validity Checker. This is where all collision checking will be done.
        ss.setStateValidityChecker(validity_checker)

        start = ob.State(chain)
        goal = ob.State(chain)
        
        chain.setup()
        
        # Following lines assign random values to start and goal states according to the states.
        # start.random()
        # goal.random()
        
        
        for i in range(numLinks):
            start[i] = 0.
        for i in range(numLinks):
            goal[i] = 0.
        goal[0] = math.pi

        print("Start:",start)
        print("Goal:",goal)

        
        
        # Set the start and goal states in the state space.
        ss.setStartAndGoalStates(start, goal)

        si = ss.getSpaceInformation()
        
        if current_planner == 'RRT':
            planner = og.RRTConnect(si)
            planner.setRange(2.8)   # Set the range of RRTConnect
        
        elif current_planner == 'KPIECE':
            planner = og.KPIECE1(si)    # Change Planner Here
        
        
        ss.setPlanner(planner)

        solved = ss.solve(5.0)      # Solve the problem
        # print(planner.getRange())
        # if solved:
        #     # try to shorten the path
        #     # ss.simplifySolution()     # this will automatically shorten the path. Uncomment if needed.
        #     # print the simplified path
        #     print(ss.getSolutionPath())
        
        
        # For KPIECE, following are default values:
        # Goal Bias: 0.05
        # Range: 3.55 (For Current Map)
        # Min Valid Path Fraction: 0.2
        # Failed Expansion Score Factor: 0.5

        
        

        if benchmarking == 'ON':
            runtime_limit = 5
            memory_limit = 8192
            run_count = 10
            request = ot.Benchmark.Request(runtime_limit, memory_limit, run_count, 0.5)
            b = ot.Benchmark(ss, "KinematicChain")
            b.addExperimentParameter("num_links", "INTEGER", str(numLinks))
            Range = np.linspace(0.01,0.7,35).tolist()
            for i in Range:
                planner = og.KPIECE1(si)
                planner.setGoalBias(i)
                b.addPlanner(planner)

            b.benchmark(request)
            b.saveResultsToFile(f"Map_{Map_Num}_{numLinks}_{current_planner}.log")

            Command = 'ompl_benchmark_statistics.py Map_'+str(Map_Num)+'_'+str(numLinks)+'_'+str(current_planner)+'.log -d database_'+str(Map_Num)+'.db'
            os.system(Command)

            # db = "./benchmark.db"
            # vis = omplVisualize.BenchmarkVisualize(db)
            # vis.box_plot('time','plannerid')