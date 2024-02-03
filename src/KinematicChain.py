import ompl.base as ob
import ompl.geometric as og
from ompl import util as ou
from ompl import tools as ot

import KinematicHeader
from KinematicHeader import createTestEnvironment
from KinematicHeader import KinematicChainValidityChecker
from KinematicHeader import KinematicChainSpace

import os


if __name__ == "__main__":
    import math
    '''
    # Info:
    # si: Space Infomation
    # env: Environment to be used. Contains the map to be used. Contains segments representing boundary and obstacles.
    # numlinks: The number of links present in robot
    # ValidityChecker: Collision Check is implemented here. This checks if the randomly sampled state is valid.
    '''

    numLinks = int(5)   # change number of links here

    # env = createHornEnvironment(d=numLinks, eps=math.log(float(numLinks)) / float(numLinks))
    env = createTestEnvironment(d=numLinks)
    # change link-length here
    chain = KinematicChainSpace(num_links=numLinks, link_length=1.0 / float(numLinks), env=env, proj_chain_count = 3)

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
    goal[0] = math.pi/2

    print("Start:",start)
    print("Goal:",goal)


    # Set the start and goal states in the state space.
    ss.setStartAndGoalStates(start, goal)

    si = ss.getSpaceInformation()
    planner = og.KPIECE1(si)    # Change Planner Here
    # planner = og.RRTConnect(si)
    # planner.setRange(1)   # Set the range of RRTConnect
    
    ss.setPlanner(planner)

    solved = ss.solve(1.0)      # Solve the problem

    if solved:
        # try to shorten the path
        # ss.simplifySolution()     # this will automatically shorten the path. Uncomment if needed.
        # print the simplified path
        print(ss.getSolutionPath())
    

    runtime_limit = 60.0
    memory_limit = 1024
    run_count = 20
    request = ot.Benchmark.Request(runtime_limit, memory_limit, run_count, 0.5)
    b = ot.Benchmark(ss, "KinematicChain")
    b.addExperimentParameter("num_links", "INTEGER", str(numLinks))
    # Ranges = [7,10,20,30,40]
    # for i in Ranges:
    #     planner = og.RRTConnect(si)
    #     planner.setRange(i)
    #     b.addPlanner(planner)
    b.addPlanner(planner)
    # planners = [
    #     # og.STRIDE(ss.getSpaceInformation()),
    #     og.EST(ss.getSpaceInformation()),
    #     # og.KPIECE1(ss.getSpaceInformation()),
    #     og.RRT(ss.getSpaceInformation()),
    #     og.PRM(ss.getSpaceInformation())
    # ]

    # for planner in planners:
    #     b.addPlanner(planner)

    b.benchmark(request)
    b.saveResultsToFile(f"kinematic_{numLinks}.log")

    Command = 'ompl_benchmark_statistics.py -a kinematic_'+str(numLinks)+'.log'
    os.system(Command)
