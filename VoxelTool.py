'''
Writen by Grayson Frazier

This script contains the "VoxelClick" class which allows for interaction with plot
To be used with plot_mic_patches function in MicFileTool
Newly Updated with Grayson's hopeful attempt to reintroduce borders
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import Collection
from matplotlib.patches import Polygon
from matplotlib.patches import Patch
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection
import numpy as np


class Voxel():
    '''
    This will be used to efficiently keep track of the different plotted voxels
    Includes the actual voxels as well as the borderlines
    '''
    def __init__(self, shape, border):
        self.shape = shape
        self.border = border
        #use left, right, and middle borders

class VoxelClick():
    '''
    Contains all the Voxels from Mic Data
    Organizes the Voxels for efficiency
    Creates an interactive environment

    Legacy File Format:
        Col 0-2 x, y, z
        Col 3   1 = triangle pointing up, 2 = triangle pointing down
        Col 4 generation number; triangle size = sidewidth /(2^generation number )
        Col 5 Phase - 1 = exist, 0 = not fitted
        Col 6-8 orientation
        Col 9  Confidence
    '''
    def __init__(self, fig, snp, sw, Mic):
        '''
        Canvas figure, snp data, base sidewdth
        '''
        self.snp = snp #all voxel data in legacy format
        self.sw = sw #base generation for triangles
        self.fig = fig
        self.size = self.sw/2**(snp[1,4])
        self.mic = Mic

    def press(self, event):
        if event.key == None:
            return
        if event.key == "enter":
            assert self.clicked_angles != [], "Please click a voxel first"
            self.mic.plot_mic_patches(plotType=1,minConfidence=0.8,maxConfidence=1.0,limitang=True,angles=self.clicked_angles)

    def onclick(self, event):
        if event.xdata == None or event.ydata == None:
            return #ensures the mouse event is on the canvas

        xdata = event.xdata
        ydata = event.ydata

        self.centers = []
        for i in range(len(self.snp)):
            if self.snp[i,3] == 1: #for a triangle pointed up
                self.centers.append([self.snp[i,0]+self.sw/(2**self.snp[i,4])/2 , self.snp[i,1]+self.sw/(2**self.snp[i,4])/2*np.sqrt(3)/2])
            else:
                self.centers.append([self.snp[i,0]+self.sw/2**self.snp[i,4]/2 , self.snp[i,1]-self.sw/(2**self.snp[i,4])/2*np.sqrt(3)/2])
        self.centers = np.array(self.centers) #this is used to place the marker in the center of the triangluar voxel

        def find_indices(x,y):
            closest_x_distance = abs(self.centers[1,0]-x)
            closest_y_distance = abs(self.centers[1,1]-y)
            for i in range(len(self.snp)):
                if abs(self.centers[i,0] - x) < closest_x_distance:
                    closest_x_distance = abs(self.centers[i,0]-x)
                    closest_index = i
                if abs(self.centers[i,1] - y) < closest_y_distance:
                    closest_y_distance = abs(self.centers[i,1]-y)

            indices = [] #should just be one indice, but will include a list to avoid error
            for i in range(len(self.snp)):
                if abs(abs(self.centers[i,0]-x)-closest_x_distance) <= closest_x_distance:
                    if abs(abs(self.centers[i,1]-y)-closest_y_distance) <= closest_y_distance:
                        indices.append(i)
            return indices

        indices = find_indices(xdata, ydata)
        assert len(indices) != 0, "No Indices Found"

        def list_average(L): #just to find the average orientation
            assert type(L) == list
            sum = 0
            for i in L:
                sum += i
            return sum/len(L)

        Orientation1 = [self.snp[i, 6] for i in indices]
        Orientation2 = [self.snp[i, 7] for i in indices]
        Orientation3 = [self.snp[i, 8] for i in indices]
        avg_Orientation1, avg_Orientation2, avg_Orientation3 = list_average(Orientation1), list_average(Orientation2), list_average(Orientation3)

        self.clicked_angles = [avg_Orientation1, avg_Orientation2, avg_Orientation3]

        if len(indices)>1:
            print("------------------------------------------------------\nAverage Euler Angles: (", self.clicked_angles[0], ",", self.clicked_angles[1], ",", self.clicked_angles[2], ")")
            print(self.snp[indices[0]][9])
            print("To replot, press enter")
        else:
            print("------------------------------------------------------\nEuler Angles: (",self.clicked_angles[0], ",", self.clicked_angles[1], ",", self.clicked_angles[2], ")")
            print(self.snp[indices[0]][9])
            print("To replot, press enter")

        #self.mic.plot_mic_patches(plotType=1,minConfidence=0.8,maxConfidence=1.0,limitang=True,angles=self.clicked_angles)
        if event.dblclick: #if event.dblclick:#double click to replot the grain
            print("gonna plot now")
            self.mic.plot_mic_patches(plotType=1,minConfidence=0,maxConfidence=1,limitang=True,angles=self.clicked_angles)

    def connect(self):
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.press)

#################################################################################################
########### STUFF TO DO WITH BORDERS....#########################################################
#################################################################################################

def order_snp(snp):
    '''
    Takes the points in snp data and orders it
    Left to Right and then Top to Bottom (like reading)
    This is already assumed in the files, but just to be safe :)
    Assumes all z are the same
    '''
    snp = sorted(snp , key=lambda k: [k[1], k[0], k[3]]) #also sorts by triangle orientation (row, column, then orientation)
    return snp

def Make_Borders(snp, sw):
    '''
    Requires the snp data and side width
    Parses through the entire mic file
    Makes borders to later be utilized by Borders class
    Left/Upper Voxel Data and then Right/Lower Voxel Data
    '''

    #MAKE TUPLES (ROW, u/d)  [u/d is just 'u' or 'd']
    order_snp(snp) #just to be sure
    #initializing primary values for the loop
    row_dict = {} #the format will be row:elements
    ################################
    ###Making Some changes##########
    ################################
    
    
    #initialize values
    y_value = snp[0,0] #initial value
    if snp[0][3] == 1: #Pointing up
        row = (0,'u')
    else:
        row = (0,'d')
    row_dict[row] = []
    
    #making rows to later make a border
    for i in range(len(snp)):
        if abs(snp[i,1] - y_value)<= .000001: #agrees with the accuracy of most mic files
            if snp[i,3] == 1: #Triangle facing up
               	 np.append(row_dict[(row[0], "u")], snp[i])

            else:
                row_dict[(row[0], "d")].append(snp[i])
        else:
            row = (row[0]+1, 'u')###change here
            y_value = snp[i,1]
            row_dict[row] = snp[i] #start a new row term
    print("here boi:", row_dict[list(row_dict.keys())[0]])
    print("here boi:", row_dict[(0, 'u')])
    #list(x.keys()) or do for i in x.keys()
    '''
    row_i = 1#initial value
    #row_dict[row] = []
    #making rows to later make a border
    for i in range(len(snp)):
        if abs(snp[i,1] - y_value)<= .000001: #agrees with the accuracy of most mic files
            if snp[i,3] == 1: #Triangle facing up
                row_dict[(row_i, "u")].append(snp[i])
            else:
                row_dict[(row_i, "d")].append(snp[i])
        else:
            row_i += 1 #(row+1, 'u')###change here
            x_value = snp[i,0]
            row_dict[(row_i, 'u'] = snp[i] #start a new row term
    #list(x.keys()) or do for i in x.keys()
    '''

    #and now it's time to make da borders!  Go top to bottom
    '''
    The thing below is kinda important...
    '''
    border_list = np.array([]) #in the form [line segment, left/up voxel, right/down voxel]
    outside_edges = np.array([]) #these are just outside borders in the form [line segment, voxel]

    """The Row-to-Row Borders""" #make sure last one doesn't call an error
    for row in row_dict.keys():
        if row[1] == 'd': #dealing with down triangles"range((len(row_dict.keys())-len(row_dict.keys())%2)/2): #for every row (accounts for up and down hence the modulus)" <-- nevermid, ditched this format (too much work)
            for row1indx in range(len(row_dict[row])): #the row1indx is just the index of that row, just to make it easier for Grayson
                for row2indx in range(len(row_dict[(row[0]+1, "u")])): #testing points in next row which are up triangless
                    bottom_key = (row[0] +1, "u") #just for ease on the eyes
                    if abs(row_dict[bottom_key][row2indx][0]-row_dict[row][row1indx][0]) <= 1.01*sw/2: #if the two points are within the desired side width (think right triangles)
                        if row_dict[row][row1indx][0] - row_dict[bottom_key][row2indx][0] <= 0: #the bottom row's point is to the left
                            if row1indx != 0: #the point is not the first in the index
                                border_list = np.append(border_list, [ [row_dict[row][row1indx][0:2], row_dict[bottom_key][row2indx][0:2]] , row_dict[row][row1indx-1], row_dict[bottom_key][row2indx]])
                                #                                      ^Line Segment                                                         ^Left/Upper Voxel          ^Lower/Right Voxel
                            else: #if first index, it is a left edge
                                outside_edges = np.append(outside_edges, [ [row_dict[row][row1indx][0:2], row_dict[bottom_key][row2indx][0:2]], row_dict[bottom_key][row2indx]])
                        else: #bottom row index to the right
                            if row2indx != len(row_dict[row+1])-1: #the point is not the last in the index
                                border_list = np.append(border_list, [ [row_dict[row][row1indx][0:2], row_dict[bottom_key][row2indx][0:2]] , row_dict[bottom_key][row2indx-1], row_dict[row][row1indx]])
                            else: #iif last index, it is a right edge
                                outside_edges = np.append(outside_edges, [ [row_dict[row][row1indx][0:2], row_dict[bottom_key][row2indx][0:2]] , row_dict[bottom_key][row2indx-1]])
                        break #break to save computing time

    """The Same-Row (i.e. horizontal) Borders"""
    #print row_dict[row_dict.keys[0]]
    for row in row_dict.keys(): #must fix 'u'
        if row[1] == 'u':
            for i in range(len(row_dict[row])-1): #-1 to account for the last edge
                print("Check_0" + str(i))
                print(row_dict[row])
                print(row_dict[row][i])
                x = row_dict[row]
                y = x[i]
                print("y is ", y)
                z = y[0:2]
                print("Check 1: " + str(row_dict[row][i][0:2]))
                print("Check 2: " + str(row_dict[row][i][0:2]))
                print("Check 3: " + str(row_dict[row][i]))
                print("Check 4: " + str(row_dict[(row[0], "d")][i]))

                border_list = np.append(border_list, [ [row_dict[row][i][0:2],        row_dict[row][i][0:2]],                 row_dict[row][i],         row_dict[(row[0], "d")][i]])
                #border_list = np.append(border_list, [ [row_dict[row][i][0:2],        row_dict[row][i][0:2]],                row_dict[row][(row[0], "d")]])
                border_list = np.append(border_list, [ [row_dict[row][row1indx][0:2], row_dict[bottom_key][row2indx][0:2]] , row_dict[row][row1indx-1], row_dict[bottom_key][row2indx]])
                #######Something went wrong here.................................................................
                #based on border_list = np.append(border_list, [ [row_dict[row][row1indx][0:2], row_dict[bottom_key][row2indx][0:2]] , row_dict[row][row1indx-1], row_dict[bottom_key][row2indx]])

    """The Top and Bottom Edges"""
    if (0, "u") in row_dict.keys():
        for i in range(len(row_dict[row])):
            point = row_dict[(0, "u")][i][0:2]
            sl = row_dict[(0, "u")][i][4] #generation number
            left_line =  [ [point, [point[0] + sl/2, point[1] + sl*sqrt(3)/2]] ]
            right_line = [ [[point[0] + sl/2, point[1] + sl*sqrt(3)/2]], [point[0] + sl, point[1]] ]

            outside_edges = np.append(outside_edges, [ left_line, row_dict[(0, "u")][i]])
            outside_edges = np.append(outside_edges, [ right_line, row_dict[(0, "u")][i]])

    blah = [] #sorry, coulldn't think of a name
    for term in row_dict.keys():
        blah.append(term[0]) #row number
    if (max(blah), "d") in row_dict.keys():
        point = row_dict[(max(blah), "d")][i][0:2]
        sl = row_dict[(max(blah), "d")][i][4] #generation number
        left_line =  [ [point, [point[0] + sl/2, point[1] - sl*sqrt(3)/2]] ]
        right_line = [ [[point[0] + sl/2, point[1] - sl*sqrt(3)/2]], [point[0] + sl, point[1]] ]

        outside_edges = np.append(outside_edges, [ left_line, row_dict[(max(blah), "d")][i]])
        outside_edges = np.append(outside_edges, [ right_line, row_dict[(max(blah), "d")][i]])

    return border_list, outside_edges #a collection of Borders

class VoxelBorders(): #possibly make it derivative of LineCollection
    '''
    This class will highlight different borders of the voxels based on orientation
    To be used in conjuction with RotRep.py
    '''
    def __init__(self, snp, sw):
        self.snp = snp
        self.sw = sw
        self.border_list, self.outside_edges = Make_Borders(snp, sw)

        alpha_levels = []
        lines = []
        for border in self.border_list:
            lines.append(self.border_list[0])
            alpha_levels.append(1)
        self.border_collection = LineCollection(lines)
        self.alpha_levels = alpha_levels
    """
    def draw_borders(self, function):
        '''
        Function which goes through every border and will either make it visible or invisible
        Border is determined by a function inputed from the RotRep
        '''

        if function(self.LeftUpSide, self.RightDownSide)
        return None
    """
