import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3

with sqlite3.connect('/home/vishwas/DR/kinematic/dataset/Logger.db') as conn:
    c = conn.cursor()
    c.execute("select id, params, mapnum, executiontime from logs where id in (select id from logs where (mapnum, timestamp) in \
                       (select mapnum, max(timestamp) from logs where status = 1 group by mapnum)) order by mapnum")    
    data = c.fetchall()
    data = np.array(data)
    update = []
    for row in data:
        update.append([int(row[0]), eval(row[1]), int(row[2]), float(row[3])])
        # print(eval(row[1]), int(row[2]), float(row[3]))
update.pop(-4)
update.pop(6)

# Function to compare sublists
def compare_sublists(sublist1, sublist2):
    # Check if the difference between the first two elements is less than 0.1
    # print(sublist1)
    #print(abs(float(sublist1[1][1]) - float(sublist2[1][1])))
    # if abs(float(sublist1[1][1]) - float(sublist2[1][1])) < 0.2 and abs(float(sublist1[1][1]) - float(sublist2[1][1])) < 0.1:
        # Check if the third elements are equal
    # if int(sublist1[1][2]) == int(sublist2[1][2]):
    #     return 1
    if abs(float(sublist1[1][1]) - float(sublist2[1][1])) < 0.12:
        return 1
    return 0

# Creating a 2D list to store comparison results
comparison_matrix = [[0] * len(data) for _ in range(len(data))]
count = 0
d_count = 0
# Comparing each sublist with others
path_to_map = "/home/vishwas/DR/kinematic/maps_png/map_%d.png"
with open ("kinematic/learning_dataset/datapath1.txt", "w") as f:
    for i in range(len(update)):
        for j in range(i+1, len(update)):
            comparison_matrix[i][j] = compare_sublists(update[i], update[j])
            if comparison_matrix[i][j] == 1:
                count += 1
                f.write("%s %s 1\n" % (path_to_map % update[i][2], path_to_map % update[j][2]))
            else:
                d_count += 1
                f.write("%s %s 0\n" % (path_to_map % update[i][2], path_to_map % update[j][2]))
print(count)
print(d_count)
# create another txt file that contains tuple (map1, map2, 1) if the maps are similar and (map1, map2, 0) if they are not
with open("kinematic/learning_dataset/tuplelist.txt", "w") as f:
    for i in range(len(update)):
        for j in range(i+1, len(update)):
            if comparison_matrix[i][j] == 1:
                f.write("%d %d 1\n" % (update[i][2], update[j][2]))
            else:
                f.write("%d %d 0\n" % (update[i][2], update[j][2]))


