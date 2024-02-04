# Software License Agreement (BSD License)
#
#  Copyright (c) 2013, Rice University
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   * Neither the name of the Rice University nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import ompl.base as ob
import ompl.geometric as og
from ompl import util as ou
import numpy as np
import copy
import matplotlib.pyplot as plt


# from ompl.util import randomReal

import math
import boost  # You may need to install the Boost library for Python

class Segment:
    def __init__(self, p0_x, p0_y, p1_x, p1_y):
        self.x0 = p0_x
        self.y0 = p0_y
        self.x1 = p1_x
        self.y1 = p1_y

class KinematicChainProjector(ob.ProjectionEvaluator):
    def __init__(self, space, proj_chain_count, link_length):
        super().__init__(space)
        self.proj_chain_count = proj_chain_count
        self.space = space
        self.link_length = link_length

    def getDimension(self):
        return 2

    def project(self, state, projection):
        
        v = np.zeros(self.space.getDimension())
        for i in range(self.space.getDimension()):
            v[i] = state[i]
        projection[:] = self.Forward_Kinematics(v,self.link_length)
        
        # For projection using number of links
        # v = np.zeros(self.space.getDimension())
        # for i in range(self.space.getDimension()):
        #     v[i] = state[i]
        # for i in range(self.projectionMatrix.shape[0]):
        #     for j in range(self.projectionMatrix.shape[1]):
        #         if i == j:
        #             self.projectionMatrix[i, j] = 1
        # projection[:] = np.dot(v, self.projectionMatrix)
    
    def Forward_Kinematics(self, state, l):
        end_point = np.array([0,0])
        for i in state:
            end_point = end_point + np.array([l*np.cos(i),l*np.sin(i)])
            # print(end_point)
        return end_point
        

class KinematicChainSpace(ob.RealVectorStateSpace):
    def __init__(self, num_links, link_length, env=None, proj_chain_count=2):
        
        super().__init__(num_links)
        self.linkLength = link_length
        self.environment = env
        self.proj_chain_count = proj_chain_count
        bounds = ob.RealVectorBounds(num_links)
        bounds.setLow(-math.pi)
        bounds.setHigh(math.pi)
        self.setBounds(bounds)
        self.dimension = num_links
        

    def registerProjections(self):
        
        self.registerDefaultProjection(KinematicChainProjector(self, self.proj_chain_count, self.linkLength))

    def distance(self, state1, state2):

        theta1, theta2, dx, dy, dist = 0., 0., 0., 0., 0.

        for i in range(self.dimension):
            theta1 += state1[i]
            theta2 += state2[i]
            dx += math.cos(theta1) - math.cos(theta2)
            dy += math.sin(theta1) - math.sin(theta2)
            dist += math.sqrt(dx * dx + dy * dy)

        return dist * self.linkLength

    def enforceBounds(self, state):
        statet = state

        for i in range(self.dimension):
            v = statet[i] % (2.0 * math.pi)
            if v < -math.pi:
                v += 2.0 * math.pi
            elif v >= math.pi:
                v -= 2.0 * math.pi
            statet[i] = v

    def equalStates(self, state1, state2):
        flag = True
        cstate1 = state1
        cstate2 = state2

        for i in range(self.dimension):
            flag &= math.fabs(cstate1[i] - cstate2[i]) == 0         # Use Epsilon here to check if it is equal.

        return flag

    def interpolate(self, from_state, to_state, t, state):
        fromt = from_state
        tot = to_state
        statet = state

        for i in range(self.dimension):
            diff = tot[i] - fromt[i]
            if math.fabs(diff) <= math.pi:
                statet[i] = fromt[i] + diff * t
            else:
                if diff > 0.0:
                    diff = 2.0 * math.pi - diff
                else:
                    diff = -2.0 * math.pi - diff

                statet[i] = fromt[i] - diff * t
                if statet[i] > math.pi:
                    statet[i] -= 2.0 * math.pi
                elif statet[i] < -math.pi:
                    statet[i] += 2.0 * math.pi

class KinematicChainValidityChecker(ob.StateValidityChecker):
    def __init__(self, si):
        super().__init__(si)
        self.si = si

    def isValid(self, state):

        space = self.si.getStateSpace()
        s = state
        return self.isValidImpl(space, s)

    def isValidImpl(self, space, s):
        n = self.si.getStateDimension()
        segments = []
        # print("space.linkLength:",space.linkLength)
        # print("s[0]:",s[0])
        # print("s[1]:",s[1])
        # print("s[2]:",s[2])
        link_length = space.linkLength
        theta, x, y, xN, yN = 0., 0., 0., 0., 0.

        # Calculating Line segments based on angles.
        for i in range(n):
            theta += s[i]
            xN = x + math.cos(theta) * link_length
            yN = y + math.sin(theta) * link_length
            segments.append(Segment(x, y, xN, yN))
            x = xN
            y = yN

        xN = x + math.cos(theta) * 0.001
        yN = y + math.sin(theta) * 0.001
        segments.append(Segment(x, y, xN, yN))

        A = self.selfIntersectionTest(segments)
        B = self.environmentIntersectionTest(segments, space.environment)
        # In the following line, space.environment() should contain all lines of obstacles.
        return A and B

# No need to change this method.
    def selfIntersectionTest(self, env):
        for i in range(len(env)):
            for j in range(i + 1, len(env)):
                if self.intersectionTest(env[i], env[j]):
                    return False
        return True

# No need to change this method.
    def environmentIntersectionTest(self, env0, env1):
        for i in env0:
            for j in env1:
                if self.intersectionTest(i, j):
                    return False
        return True

# Need to change all epsilon values. Replace them with "<0.0001"
    def intersectionTest(self, s0, s1):
        s10_x = s0.x1 - s0.x0
        s10_y = s0.y1 - s0.y0
        s32_x = s1.x1 - s1.x0
        s32_y = s1.y1 - s1.y0
        denom = s10_x * s32_y - s32_x * s10_y

        if math.fabs(denom) < 0.0001:
            return False  # Collinear

        denom_positive = denom > 0
        s02_x = s0.x0 - s1.x0
        s02_y = s0.y0 - s1.y0
        s_numer = s10_x * s02_y - s10_y * s02_x

        if (s_numer < 0.0001) == denom_positive:
            return False  # No collision

        t_numer = s32_x * s02_y - s32_y * s02_x
        if (t_numer < 0.0001) == denom_positive:
            return False  # No collision

        if (((s_numer - denom > -0.0001) == denom_positive) or
                ((t_numer - denom > 0.0001) == denom_positive)):
            return False  # No collision

        return True

def createHornEnvironment(d, eps):
    env_file = open(f"environment_{d}.dat", "w")
    env = []
    w = 1.0 / float(d)
    x = w
    y = -eps
    xN, yN = 0., 0.
    theta = 0.
    scale = w * (1.0 + math.pi * eps)

    env_file.write(f"{x} {y}\n")
    for i in range(d - 1):
        theta += math.pi / float(d)
        xN = x + math.cos(theta) * scale
        yN = y + math.sin(theta) * scale
        env.append(Segment(x, y, xN, yN))
        x = xN
        y = yN
        env_file.write(f"{x} {y}\n")

    theta = 0.
    x = w
    y = eps
    env_file.write(f"{x} {y}\n")
    scale = w * (1.0 - math.pi * eps)
    for i in range(d - 1):
        theta += math.pi / float(d)
        xN = x + math.cos(theta) * scale
        yN = y + math.sin(theta) * scale
        env.append(Segment(x, y, xN, yN))
        x = xN
        y = yN
        env_file.write(f"{x} {y}\n")

    env_file.close()
    return env

def createEmptyEnvironment(d):
    env_file = open(f"environment_{d}.dat", "w")
    env = []
    env_file.close()
    return env

def createTestEnvironment(d):
    env_file = open(f"environment_{d}.dat", "w")
    env = []
    plt.figure()

    Border_value = 1 + (2/d)
    x = -Border_value
    y = -Border_value
    xN = x
    yN = y + 2*Border_value
    env.append(Segment(x, y, xN, yN))
    env_file.write(f"{x} {y}\n")
    plt.plot([x, xN], [y, yN], 'r-')
    x = xN
    y = yN
    xN = x + 2*Border_value
    yN = y
    env.append(Segment(x, y, xN, yN))
    env_file.write(f"{x} {y}\n")
    plt.plot([x, xN], [y, yN], 'r-')
    x = xN
    y = yN
    xN = x
    yN = y - 2*Border_value
    env.append(Segment(x, y, xN, yN))
    env_file.write(f"{x} {y}\n")
    plt.plot([x, xN], [y, yN], 'r-')
    x = xN
    y = yN
    xN = x - 2*Border_value
    yN = y
    env.append(Segment(x, y, xN, yN))
    env_file.write(f"{x} {y}\n")
    plt.plot([x, xN], [y, yN], 'r-')
    

    Obstacle_value  = 2/d
    x = 1-Obstacle_value
    y = 1-Obstacle_value
    xN = x
    yN = y + Obstacle_value
    env.append(Segment(x, y, xN, yN))
    env_file.write(f"{x} {y}\n")
    plt.plot([x, xN], [y, yN], 'b-')
    x = xN
    y = yN
    xN = x + Obstacle_value
    yN = y
    env.append(Segment(x, y, xN, yN))
    env_file.write(f"{x} {y}\n")
    plt.plot([x, xN], [y, yN], 'b-')
    x = xN
    y = yN
    xN = x
    yN = y - Obstacle_value
    env.append(Segment(x, y, xN, yN))
    env_file.write(f"{x} {y}\n")
    plt.plot([x, xN], [y, yN], 'b-')
    x = xN
    y = yN
    xN = x - Obstacle_value
    yN = y
    env.append(Segment(x, y, xN, yN))
    env_file.write(f"{x} {y}\n")
    plt.plot([x, xN], [y, yN], 'b-')

    env_file.close()
    # plt.show()

    return env