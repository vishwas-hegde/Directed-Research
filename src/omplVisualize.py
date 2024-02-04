import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon 
import sqlite3
import math

class BenchmarkVisualize():
    def __init__(self, db_path):
        self.db = db_path
        self.read_db(self.db)
        
    
    def read_db(self):
        with sqlite3.connect(self.db) as conn:
            query = "Select * from runs"
            self.df = pd.read_sql_query(query, conn)

        return self.df
    
    def line_plot(self):
        pass

    def box_plot(self, col, by):
        self.df.boxplot(column=col, by=by, vert=True)
        plt.ylabel(col)
        plt.xlabel(by) 
        plt.title('Boxplot of %s by %s'%(col,by))
        plt.suptitle('')
        plt.show()
        pass


class ResultVisualize():
    def __init__(self, start, goal, results, obstacles,length, base):
        self.start = start 
        self.goal = goal
        self.solution = results
        self.base = base
        self.obstacles = obstacles
        self.link_lengths = length

    def plot(self, return_ax=False):
        # Create empty map
        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(111)

        # Set the axis limits based on the map size
        ax.set_xlim(-1.66, 1.66)
        ax.set_ylim(-1.66, 1.66)
        # Set labels and title
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Map Visualization')

        # Draw each obstacle as a polygon
        # for obstacle in obstacles:
        #     polygon = Polygon(
        #         obstacle, closed=True, edgecolor='black', facecolor='gray'
        #     )
        #     ax.add_patch(polygon)
        for obstacle in self.obstacles:
            x, y, xN, yN = obstacle
            ax.plot([x, xN], [y, yN], color='black') 

        if self.solution != []:
            # draw path
            for i in range(len(self.solution)-1):
                # draw robot
                for c in self.interpolate(self.solution[i], self.solution[i+1], num=10): 
                    self.draw_robot(ax, c , edgecolor="grey")
            for i in range(len(self.solution)):
                # if i == len(self.solution) - 1:
                #     continue
                # p1 = self.forward_kinematics(self.solution[i])
                # p2 = self.forward_kinematics(self.solution[i + 1])
                # ax.plot(
                #     [p1[0], p2[0]],
                #     [p1[1], p2[1]],
                #     color="y",
                #     linewidth=1,
                # )
                self.draw_robot(ax, self.solution[i], edgecolor="blue")
                
        if self.start is not None and self.goal is not None:
            self.draw_robot(ax, self.start, edgecolor="red")
            self.draw_robot(ax, self.goal, edgecolor="green")
        plt.show()
        # if return_ax:
        #     return ax, fig
        # else:
        #     plt.show()
    
    def draw_robot(self, ax, config, edgecolor="b", facecolor="black"):
        # compute joint positions and draw lines
        positions = self.forward_kinematics(config)
        # Draw lines between each joint
        for i in range(len(positions) - 1):
            line = np.array([positions[i], positions[i + 1]])
            ax.plot(line[:, 0], line[:, 1], color=edgecolor)
        # Draw joint
        for i in range(len(positions)):
            ax.scatter(positions[i][0], positions[i][1], s=5, c=facecolor)

    def forward_kinematics(self, config):
        """Compute the joint coordinates given a configuration of joint angles.
        The last endpoint would be used for visualization of the sampling
        arguments:
            config: A list of joint angles in radians.

        return:
            edges: A list of joint coordinates.
        """
        # Initialize the starting point as the fixed base
        joint_positions = [self.base]
        start_point = np.array(self.base)
        angle = 0

        x, y = start_point[0], start_point[1]  # Initial joint position

        for l, conf_angle in zip(self.link_lengths, config):
            angle += conf_angle
            x += l * math.cos(angle)
            y += l * math.sin(angle)

            joint_positions.append([x, y])

        return joint_positions
    
    def interpolate(self, config1, config2, num=10):
        """Interpolate between two configurations
        arguments:
            p1 - config1, [joint1, joint2, joint3, ..., jointn]
            p2 - config2, [joint1, joint2, joint3, ..., jointn]

        return:
            A Euclidean distance in 
            list with num number of configs from linear interploation in S^1 x S^1 x ... x S^1 space.
        """

        ### YOUR CODE HERE ###
        interpolated_configs = [config1]

        # Interpolate between config1 and config2
        for i in range(1, num):
            interpolated_step = [self.interpolate_angle(j1, j2, num)[i] for j1, j2 in zip(config1, config2)]
            interpolated_configs.append(interpolated_step)

        return interpolated_configs
    
    def interpolate_angle(self, angle1, angle2, num):
        """Interpolate between two angles"""
        # Calculate the step size from angle1 to angle2
        step_size = self.angle_diff(angle1, angle2, absolute=False) / (num - 1)

        # Interpolate
        interpolated_angles = [
            self.wrap_to_pi(angle1 + i * step_size) 
            for i in range(num)
        ]

        return interpolated_angles
    
    def wrap_to_pi(self, angle):
        """Wrap an angle to [-pi, pi]"""
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return angle
    
    def angle_diff(self, angle1, angle2, absolute=True):
        """Calculate the min difference between two angles ranged in [-pi, pi]
        arguments:
            angle1: from angle1
            angle2: to angle2
            abs: if return the absolute value of the difference,
                if so, the result is always positive and ranges in [0, pi]
                else, it will return the signed difference from angle1 to angle2
        """
        angle_diff = angle2 - angle1

        # Calculate the absolute difference
        min_diff = np.min(
            [np.abs(angle_diff),
            2 * np.pi - np.abs(angle_diff)]
        )
        if absolute:
            return min_diff

        # Determine if the difference is
        # in the positive or negative direction
        is_pos = angle_diff % (2 * np.pi) < np.pi
        if not is_pos:
            min_diff = -min_diff

        return min_diff

        

