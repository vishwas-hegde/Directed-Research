import matplotlib.pyplot as plt

class Segment:
    def __init__(self, p0_x, p0_y, p1_x, p1_y):
        self.x0 = p0_x
        self.y0 = p0_y
        self.x1 = p1_x
        self.y1 = p1_y

def Line(start_point, length, axis):
    x = start_point[0]
    y = start_point[1]
    if not axis:
        xN = x + length
        yN = y
    elif axis:
        xN = x
        yN = y + length
    return Segment(x, y, xN, yN), [x, y, xN, yN]

def Square(StartPoint, length):
    env = []
    New_Point = StartPoint
    point_List = []
    axis = False
    Flag = False
    # plt.figure()
    for i in range(4):
        if Flag == True:
            length = -length
            Flag = False
        axis = not axis
        Line_Segment, Points = Line(New_Point, length, axis)
        New_Point = [Points[2],Points[3]]
        point_List.append(Points)
        env.append(Line_Segment)
        # plt.plot([Old_Point[0], New_Point[0]], [Old_Point[1], New_Point[1]], 'r-')
        if i == 1:
            Flag = True
    # plt.show()
    return env, point_List

def createTestEnvironment(d):
    env_file = open(f"environment_{d}.dat", "w")
    env = []
    plt.figure()

    Border_value = 1 + (2/d)
    x = -Border_value
    y = -Border_value
    Segments, Points = Square([x,y],2*Border_value)

    env.extend(Segments)
    for point in Points:
        plt.plot([point[0], point[2]], [point[1], point[3]], 'r-')

    Segments, Points = Square([-0.5,0.5],0.3)
    env.extend(Segments)
    for point in Points:
        plt.plot([point[0], point[2]], [point[1], point[3]], 'r-')

    Segments, Points = Square([-0.5,-0.5],0.3)
    env.extend(Segments)
    for point in Points:
        plt.plot([point[0], point[2]], [point[1], point[3]], 'r-')

    Segments, Points = Square([0.2,-0.2],0.15)
    env.extend(Segments)
    for point in Points:
        plt.plot([point[0], point[2]], [point[1], point[3]], 'r-')
    
    Segments, Points = Square([0.45, 0.05],0.5)
    env.extend(Segments)
    for point in Points:
        plt.plot([point[0], point[2]], [point[1], point[3]], 'r-')

    plt.plot([0,1],[0,0])
    plt.plot([0,0],[0,1])
    env_file.close()
    plt.show()

    return env

def ReadEnvironmentFromFile(Num):
    # env_file = open(f"environment_{d}.dat", "r")
    path = '/home/dhrumil/WPI/Sem2/DR/Map/maps_txt/map_' + str(Num) + '.txt'
    file = open(path,'r')
    Data = eval(file.read())
    env = []
    for line in Data:
        x0, y0, x1, y1 = line
        env.append(Segment(x0, y0, x1, y1))
    return env

if __name__ == "__main__":
    a = ReadEnvironmentFromFile(0)