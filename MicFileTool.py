'''
Writen by He Liu
Wed Apr 26 2017
This script will contains the basic tool for reading mic file and plot them.


Modified by Doyee Byun & Grayzon Frazier
2018

Modifications for coloring made by Doyee Byun
Including References to VoxelTool written by Grayson Frazier
April 10, 2018

Added functionality for square matrix data format,
working on gui elements.
July 14, 2018

Now modified to be fully interactive!
July 24, 2018
'''
import numpy as np
import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.collections import PolyCollection
import RotRep
from VoxelTool import VoxelClick
from VoxelTool import SquareVoxelClick

class SquarePoint():
    def __init__(self, xi, yi):
        self.x = xi
        self.y = yi
        self.coords = (xi,yi)
        self.up = (xi,yi+1)
        self.right = (xi+1,yi)
        self.down = (xi,yi-1)
        self.left = (xi-1,yi)
        self.up_blocked = False
        self.right_blocked = False
        self.down_blocked = False
        self.left_blocked = False
        self.angles = [0.0,0.0,0.0]
        return

    def block_up(self):
        self.up = None
        self.up_blocked = True
        return

    def block_right(self):
        self.right = None
        self.right_blocked = True
        return

    def block_down(self):
        self.down = None
        self.down_blocked = True
        return

    def block_left(self):
        self.left = None
        self.left_blocked = True
        return

    def set_angles(self, angles):
        self.angles = angles

    def check_points(self,data,xlim,ylim,misor_thresh):
        upx,upy = self.up
        rightx,righty = self.right
        downx,downy = self.down
        leftx,lefty = self.left
        if upx < 0 or upx > xlim or upy < 0 or upy > ylim:
            self.block_up()
        if rightx < 0 or rightx > xlim or righty < 0 or righty > ylim:
            self.block_right()
        if downx < 0 or downx > xlim or downy < 0 or downy > ylim:
            self.block_down()
        if leftx < 0 or leftx > xlim or lefty < 0 or lefty > ylim:
            self.block_left()

        angles = np.array([self.angles])
        if not self.up_blocked:
            upangles = np.array([[data[upx,upy,3],data[upx,upy,4],data[upx,upy,5]]])
            up_misor = RotRep.MisorinEulerZXZ(angles,upangles)
            if up_misor > misor_thresh:
                self.block_up()
        if not self.right_blocked:
            rightangles = np.array([[data[rightx,righty,3],data[rightx,righty,4],data[rightx,righty,5]]])
            right_misor = RotRep.MisorinEulerZXZ(angles,rightangles)
            if right_misor > misor_thresh:
                self.block_right()
        if not self.down_blocked:
            downangles = np.array([[data[downx,downy,3],data[downx,downy,4],data[downx,downy,5]]])
            down_misor = RotRep.MisorinEulerZXZ(angles,downangles)
            if down_misor > misor_thresh:
                self.block_down()
        if not self.left_blocked:
            leftangles = np.array([[data[leftx,lefty,3],data[leftx,lefty,4],data[leftx,lefty,5]]])
            left_misor = RotRep.MisorinEulerZXZ(angles,leftangles)
            if left_misor > misor_thresh:
                self.block_left()


def dist_to_line(point,line):
    '''

    :param point: array,size=[1,2]
    :param line: array, size = [2,2]
    :return:
    '''
    point = np.array(point)
    line = np.array(line)
    r_1 = point-line[0,:]
    r_2 = line[0,:]-line[1,:]
    #print r_1,r_2
    dist = np.linalg.norm(r_1)*np.sqrt(1-(np.abs(np.sum(r_1*r_2))/(np.linalg.norm(r_1)*np.linalg.norm(r_2)))**2)
    return dist

def select_line_mic(snp):
    '''
    select mic along a line
    :param snp:
    :return:
    '''
    line = np.array([[0,0.24],[0.22,0.13]])
    d = 0.02
    N = snp.shape[0]
    bool_lst = [False]*N
    for i in range(N):
        dist = dist_to_line(snp[i,0:2],line)
        if dist < d:
            bool_lst[i] = True
    new_snp = snp[bool_lst,:]
    plt.plot(line[:,0],line[:,1])
    return new_snp

def save_mic_file(fname,snp,sw):
    '''
    save to mic file
    :param fname:
    :param snp:
    :param sw:
    :return:
    '''
    # target = open(fname, 'w')
    # target.write(str(sw))
    # target.write('\n')
    # target.close()
    np.savetxt(fname,snp,delimiter=' ',fmt='%f',header=str(sw),comments='')


def read_mic_file(fname):
    '''
    this will read the mic file
      %%
      %% Legacy File Format:
      %% Col 0-2 x, y, z
      %% Col 3   1 = triangle pointing up, 2 = triangle pointing down
      %% Col 4 generation number; triangle size = sidewidth /(2^generation number )
      %% Col 5 Phase - 1 = exist, 0 = not fitted
      %% Col 6-8 orientation
      %% Col 9  Confidence
      %%
    :param fname:
    :return:
        sw: float, the side width
        snp: [n_voxel,n_feature] numpy array

    '''
    with open(fname) as f:
        content = f.readlines()
    print(content[1])
    print(type(content[1]))
    sw = float(content[0])
    try:
        snp = np.array([[float(i) for i in s.split(' ')] for s in content[1:]])
    except ValueError:
        try:
            snp = np.array([[float(i) for i in s.split('\t')] for s in content[1:]])
        except ValueError:
            print('unknown deliminater')

    print('sw is {0} \n'.format(sw))
    print('shape of snp is {0}'.format(snp.shape))
    return sw,snp

    # snp = pd.read_csv(filename,delim_whitespace=True,skiprows=0,header=0).values
    # sw =  pd.read_csv(filename,delim_whitespace=True,nrows=1,header=0).values
    # print snp
    # print(snp.shape)
    # print sw


def plot_mic(snp,sw,plotType,minConfidence,maxConfidence,scattersize=2):
    '''
    plot the mic file
    :param snp:
    :param sw:
    :param plotType:
    :param minConfidence:
    :return:
    '''
    snp = snp[minConfidence<=snp[:,9]<=maxConfidence,:]
    N = snp.shape[0]
    mat = np.empty([N,3,3])
    quat = np.empty([N,4])
    rod = np.empty([N,3])
    if plotType==2:
        fig, ax = plt.subplots()
        sc = ax.scatter(snp[:,0],snp[:,1],c=snp[:,9],cmap='cool')
        plt.colorbar(sc)
        plt.show()
    if plotType==3:
        print('h')
        for i in range(N):
            mat[i,:,:] = RotRep.EulerZXZ2Mat(snp[i,6:9]/180*np.pi)
            #print mat[i,:,:]
            quat[i,:] = RotRep.quaternion_from_matrix(mat[i,:,:])
            #print quat[i,:]
            rod[i,:] = RotRep.rod_from_quaternion(quat[i,:])
        #print(rod)
        fig, ax = plt.subplots()
        ax.scatter(snp[:,0],snp[:,1],s=scattersize,facecolors=(rod+np.array([1,1,1]))/2)
        ax.axis('scaled')
        plt.show()

def recursive_limiter(x,y,data,points,coords,misor_thresh):
    new_point = None
    xi = coords[0]
    yi = coords[1]
    for i in range(0,len(points)):
        if coords[0] == points[i].x and coords[1] == points[i].y:
            new_point = points[i]
            return points
    if new_point is None:
        new_point = SquarePoint(coords[0],coords[1])
        new_angles = []
        new_angles.append(data[xi,yi,3])
        new_angles.append(data[xi,yi,4])
        new_angles.append(data[xi,yi,5])
        new_point.set_angles(new_angles)
        points.append(new_point)
    new_point.check_points(data,x,y,misor_thresh)
    if not new_point.up_blocked:
        points = recursive_limiter(x,y,data,points,[xi,yi+1],misor_thresh)
    if not new_point.right_blocked:
        points = recursive_limiter(x,y,data,points,[xi+1,yi],misor_thresh)
    if not new_point.down_blocked:
        points = recursive_limiter(x,y,data,points,[xi,yi-1],misor_thresh)
    if not new_point.left_blocked:
        points = recursive_limiter(x,y,data,points,[xi-1,yi],misor_thresh)
    return points





def square_angle_limiter(x,y, data, coords, misor_thresh=1.0):
    points = []
    xi = coords[0]
    yi = coords[1]
    angles = []
    angles.append(data[xi,yi,3])
    angles.append(data[xi,yi,4])
    angles.append(data[xi,yi,5])
    current_point = SquarePoint(xi,yi)
    current_point.set_angles(angles)
    current_point.check_points(data,x,y,misor_thresh)
    points.append(current_point)
    if not current_point.up_blocked:
        points = recursive_limiter(x,y,data,points,[xi,yi+1],misor_thresh)
    if not current_point.right_blocked:
        points = recursive_limiter(x,y,data,points,[xi+1,yi],misor_thresh)
    if not current_point.down_blocked:
        points = recursive_limiter(x,y,data,points,[xi,yi-1],misor_thresh)
    if not current_point.left_blocked:
        points = recursive_limiter(x,y,data,points,[xi-1,yi],misor_thresh)
    return points


"""
def square_angle_limiter(x,y, data ,coords,angles = [], indx=[],misor_thresh=1.0):

    #set angle limits here
    #CAUTION: MAY NOT WORK WHEN THERE IS TOO MUCH DATA
    #Recursion may become memory intensive;
    #increasing recursion limit may become necessary
    #print(misor_thresh)
    new_indx = indx
    x_index = coords[0]
    y_index = coords[1]
    if x_index < 0 or x_index >= x or y_index < 0 or y_index >= y or (x_index,y_index) in new_indx:
        return new_indx
    elif angles == []:
        angles.append(data[x_index,y_index,3])
        angles.append(data[x_index,y_index,4])
        angles.append(data[x_index,y_index,5])
    current_angs = []
    current_angs.append(data[x_index,y_index,3])
    current_angs.append(data[x_index,y_index,4])
    current_angs.append(data[x_index,y_index,5])
    current_angs = np.array([current_angs])
    old_angs = np.array([angles])
    misorient = RotRep.MisorinEulerZXZ(current_angs,old_angs)
    if misorient < misor_thresh:
        new_indx.append((x_index,y_index))
    else:
        return new_indx
    angles.clear()
    angles.append(data[x_index,y_index,3])
    angles.append(data[x_index,y_index,4])
    angles.append(data[x_index,y_index,5])
    new_indx = square_angle_limiter(x,y,data,[x_index-1,y_index],angles,new_indx,misor_thresh)
    new_indx = square_angle_limiter(x,y,data,[x_index,y_index-1],angles,new_indx,misor_thresh)
    new_indx = square_angle_limiter(x,y,data,[x_index+1,y_index],angles,new_indx,misor_thresh)
    new_indx = square_angle_limiter(x,y,data,[x_index,y_index+1],angles,new_indx,misor_thresh)
    return new_indx
"""
def set_color_range_sq(smdCopy,x,y,indx,mat,quat,rod, anglelim):
    #print("indx: ",indx)
    first = True
    for i in range(mat.shape[0]):
        yi = int(i%x)
        xi = int(i/x)
        #print(xi,yi)
        if (xi,yi) in indx or not anglelim:
            quat[i, :] = RotRep.quaternion_from_matrix(mat[i, :, :])
            rod[i, :] = RotRep.rod_from_quaternion(quat[i, :])
            if first:
                maxr = rod[i,0]
                minr = rod[i,0]
                maxg = rod[i,1]
                ming = rod[i,1]
                maxb = rod[i,2]
                minb = rod[i,2]
                maxri = i
                minri = i
                maxgi = i
                mingi = i
                maxbi = i
                minbi = i
                first = False
            else:
                if rod[i,0] > maxr:
                    maxr = rod[i,0]
                    maxri = i
                elif rod[i,0] < minr:
                    minr = rod[i,0]
                    minri = i
                if rod[i,1] > maxg:
                    maxg = rod[i,1]
                    maxgi = i
                elif rod[i,1] < ming:
                    ming = rod[i,1]
                    mingi = i
                if rod[i,2] > maxb:
                    maxb = rod[i,2]
                    maxbi = i
                elif rod[i,2] < minb:
                    minb = rod[i,2]
                    minbi = i
        else:
            rod[i,:]=[0.0,0.0,0.0]
    maxrgb = [maxr,maxg,maxb]
    minrgb = [minr,ming,minb]
    maxangs = [rod[maxri,0],rod[maxgi,1],rod[maxbi,2]]
    minangs = [rod[minri,0],rod[mingi,1],rod[minbi,2]]
    colors = rod
    for j in range(mat.shape[0]):
        for k in range(0,3):
            colors[j,k] = (rod[j,k]-minrgb[k])/(maxrgb[k]-minrgb[k])
    return colors, maxangs, minangs

def set_color_range(mic, N, indx, mat, quat, rod):
    """
    Function for setting the color range of a plot.
    """
    first = True
    #print(indx)
    for i in range(N):
        if i in indx:
            mat[i,:,:] = RotRep.EulerZXZ2Mat(mic.snp[i,6:9]/180.0*np.pi)
            quat[i,:] = RotRep.quaternion_from_matrix(mat[i,:,:])
            rod[i,:] = RotRep.rod_from_quaternion(quat[i,:])
            if first:
                maxr = rod[i,0]
                minr = rod[i,0]
                maxg = rod[i,1]
                ming = rod[i,1]
                maxb = rod[i,2]
                minb = rod[i,2]
                maxri = i
                minri = i
                maxgi = i
                mingi = i
                maxbi = i
                minbi = i
                first = False
            else:
                if rod[i,0] > maxr:
                    maxr = rod[i,0]
                    maxri = i
                elif rod[i,0] < minr:
                    minr = rod[i,0]
                    minri = i
                if rod[i,1] > maxg:
                    maxg = rod[i,1]
                    maxgi = i
                elif rod[i,1] < ming:
                    ming = rod[i,1]
                    mingi = i
                if rod[i,2] > maxb:
                    maxb = rod[i,2]
                    maxbi = i
                elif rod[i,2] < minb:
                    minb = rod[i,2]
                    minbi = i
        else:
            rod[i,:]=[0.0,0.0,0.0]
    #print("Current rod values: ",rod)
    maxrgb = [maxr,maxg,maxb]
    minrgb = [minr,ming,minb]
    maxangs = [rod[maxri,0],rod[maxgi,1],rod[maxbi,2]]
    minangs = [rod[minri,0],rod[mingi,1],rod[minbi,2]]
    colors = rod
    for j in range(N):
        for k in range(0,3):
            colors[j,k] = (rod[j,k]-minrgb[k])/(maxrgb[k]-minrgb[k])
    return colors, maxangs, minangs

def index_from_points(points):
    indx = []
    for i in range(0,len(points)):
        xi = points[i].x
        yi = points[i].y
        indx.append((xi,yi))
    return indx

def plot_square_mic(SquareMic,squareMicData, minHitRatio,coords, misor_thresh):
    '''
    plot the square mic data
    image already inverted, x-horizontal, y-vertical, x dow to up, y: left to right
    :param squareMicData: [NVoxelX,NVoxelY,10], each Voxel conatains 10 columns:
            0-2: voxelpos [x,y,z]
            3-5: euler angle
            6: hitratio
            7: maskvalue. 0: no need for recon, 1: active recon region
            8: voxelsize
            9: additional information
    :return:
    '''
    assert(len(coords) == 0 or len(coords) == 2), "Please enter the correct number of coordinates"
    anglelim = not (coords == [])
    angles = []
    fig, ax = plt.subplots()
    indx = []
    points = []
    smdCopy = squareMicData.copy()
    (x,y,z) = squareMicData.shape
    if anglelim:
        angles.append(squareMicData[coords[0],coords[1],3])
        angles.append(squareMicData[coords[0],coords[1],4])
        angles.append(squareMicData[coords[0],coords[1],5])
        points = square_angle_limiter(x,y,smdCopy,coords,misor_thresh)
        indx = index_from_points(points)
        for i in range(0,x):
            for j in range(0,y):
                if not (i,j) in indx:
                    smdCopy[i,j,6] = 0.0
                    smdCopy[i,j,3] = angles[0]
                    smdCopy[i,j,4] = angles[1]
                    smdCopy[i,j,5] = angles[2]
    mat = RotRep.EulerZXZ2MatVectorized(smdCopy[:,:,3:6].reshape([-1,3])/180.0 *np.pi )
    quat = np.empty([mat.shape[0],4])
    rod = np.empty([mat.shape[0],3])
    colors, maxangs, minangs = set_color_range_sq(smdCopy,x,y,indx,mat,quat,rod,anglelim)
    hitRatioMask = (smdCopy[:,:,6]>minHitRatio)[:,:,np.newaxis].repeat(3,axis=2)
    #img = ((colors + np.array([1, 1, 1])) / 2).reshape([squareMicData.shape[0],squareMicData.shape[1],3]) * hitRatioMask
    img = (colors).reshape([squareMicData.shape[0],squareMicData.shape[1],3]) * hitRatioMask
    # make sure display correctly
    #img[:,:,:] = img[::-1,:,:]
    img = np.swapaxes(img,0,1)
    minX, minY = smdCopy[0,0,0:2]*1000
    maxX, maxY = smdCopy[-1,-1,0:2]*1000
    #print(minX,maxX, minY,maxY)
    if anglelim:
        (xi_first,yi_first) = indx[0]

        xmin = smdCopy[xi_first,yi_first,0]
        xmax = smdCopy[xi_first,yi_first,0]
        ymin = smdCopy[xi_first,yi_first,1]
        ymax = smdCopy[xi_first,yi_first,1]
        for (xi,yi) in indx:
            if smdCopy[xi,yi,0] <= xmin:
                xmin = smdCopy[xi,yi,0]
            if smdCopy[xi,yi,0] >= xmax:
                xmax = smdCopy[xi,yi,0]
            if smdCopy[xi,yi,1] <= ymin:
                ymin = smdCopy[xi,yi,1]
            if smdCopy[xi,yi,1] >= ymax:
                ymax = smdCopy[xi,yi,1]
        if abs(xmax-xmin) > abs(ymax-ymin):
            side_length = abs(xmax-xmin)
        else:
            side_length = abs(ymax-ymin)
        side_length = side_length / smdCopy[0,0,8] * 10
        xmin = xmin / smdCopy[0,0,8] * 10
        ymin = ymin / smdCopy[0,0,8] * 10
        ax.set_xlim([xmin -10 ,xmin + side_length + 10])
        ax.set_ylim([ymin -10 ,ymin + side_length + 10])
    print("-------------------------------------------------")
    print("MaxRod (R,G,B): ", maxangs)
    print("MinRod (R,G,B): ", minangs)
    ax.imshow(img,origin='lower',extent=[minX,maxX,minY,maxY])

    plt.title('orientation in um')
    voxels = SquareVoxelClick(fig, squareMicData,SquareMic,minHitRatio,misor_thresh)
    voxels.connect()
    plt.show()


class SquareMic():
    def __init__(self,squareMicData=None):
        self.squareMicData = squareMicData

    def load(self,fName):
        self.squareMicData = np.load(fName)

    def plot_orientation(self, coords=[], minHitRatio=0.5, misor_thresh=1.0):
        plot_square_mic(self,self.squareMicData, minHitRatio, coords, misor_thresh)

    def plot_hit_ratio(self):
        img = np.swapaxes(self.squareMicData[:,:,6], 0, 1)
        plt.imshow(img, origin='lower')
        plt.colorbar()
        plt.show()

class MicFile():
    def __init__(self,fname):
        self.sw, self.snp=self.read_mic_file(fname)
        self.color2=self.snp[:,9]
        self.bpatches=False
        self.bcolor1=False

    def read_mic_file(self,fname):
        '''
        this will read the mic file
          %%
          %% Legacy File Format:
          %% Col 0-2 x, y, z
          %% Col 3   1 = triangle pointing up, 2 = triangle pointing down
          %% Col 4 generation number; triangle size = sidewidth /(2^generation number )
          %% Col 5 Phase - 1 = exist, 0 = not fitted
          %% Col 6-8 orientation
          %% Col 9  Confidence
          %%
        :param fname:
        :return:
            sw: float, the side width
            snp: [n_voxel,n_feature] numpy array

        '''
        with open(fname) as f:
            content = f.readlines()
        print(content[1])
        print(type(content[1]))
        sw = float(content[0])
        try:
            snp = np.array([[float(i) for i in s.split(' ')] for s in content[1:]])
        except ValueError:
            try:
                snp = np.array([[float(i) for i in s.split('\t')] for s in content[1:]])
            except ValueError:
                print('unknown deliminater')

        print('sw is {0} \n'.format(sw))
        print('shape of snp is {0}'.format(snp.shape))
        return sw,snp

    def angle_limiter(self,indx, snp,angles):
        #set angle limits here
        #needs modifying for better range setting
        new_indx = []
        xl = angles[0]-1.0
        xh = angles[0]+1.0
        yl = angles[1]-1.0
        yh = angles[1]+1.0
        zl = angles[2]-1.0
        zh = angles[2]+1.0
        for i in range(0,len(indx)):
            j = indx[i]
            x = self.snp[j,6]
            y = self.snp[j,7]
            z = self.snp[j,8]
            if x > xl and x < xh and y > yl and y < yh and z > zl and z < zh:
                new_indx.append(indx[i])
        return new_indx

    def plot_mic_patches(self,plotType=1,minConfidence=0,maxConfidence=1,limitang=False,angles=[]):
        indx = []
        for i in range(0,len(self.snp)):
            if self.snp[i,9] >= minConfidence and self.snp[i,9] <= maxConfidence:
                indx.append(i)
        if limitang:
            indx = self.angle_limiter(indx,self.snp,angles)
        else:
            indx = list(range(0,len(self.snp)))
        #indx=minConfidence<=self.snp[:,9]<=maxConfidence
        minsw=self.sw/float(2**self.snp[0,4])
        tsw1=minsw*0.5
        tsw2=-minsw*0.5*3**0.5
        ntri=len(self.snp)
        if plotType==2:
            fig, ax = plt.subplots()
            if self.bpatches==False:
                xy=self.snp[:,:2]
                tmp=np.asarray([[tsw1]*ntri,(-1)**self.snp[:,3]*tsw2]).transpose()
                tris=np.asarray([[[0,0]]*ntri,[[minsw,0]]*ntri,tmp])
                self.patches=np.swapaxes(tris+xy,0,1)
                self.bpatches=True
            p=PolyCollection(self.patches[indx],cmap='viridis')
            p.set_array(self.color2[indx])
            p.set_edgecolor('face')
            ax.add_collection(p)
            ax.set_xlim([-0.6,0.6])
            ax.set_ylim([-0.6,0.6])
            fig.colorbar(p,ax=ax)
            plt.show()
        if plotType==1:
            fig, ax = plt.subplots()
            N=len(self.snp)
            mat = np.empty([N,3,3])
            quat = np.empty([N,4])
            rod = np.empty([N,3])
            if self.bcolor1==False:
                colors, maxangs, minangs = set_color_range(self, N, indx, mat, quat, rod)
                self.color1= colors
                #print("Color: ", self.color1)
                #self.bcolor1=True
            if self.bpatches==False:
                xy=self.snp[:,:2]
                tmp=np.asarray([[tsw1]*ntri,(-1)**self.snp[:,3]*tsw2]).transpose()
                tris=np.asarray([[[0,0]]*ntri,[[minsw,0]]*ntri,tmp])
                self.patches=np.swapaxes(tris+xy,0,1)
                self.bpatches=True
            p=PolyCollection(self.patches[indx],cmap='viridis')
            p.set_color(self.color1[indx])
            ax.add_collection(p)
            '''Yo future grayson, make sure interactive is a parameter!'''
            xmin = self.snp[indx[0],0]
            xmax = self.snp[indx[0],0]
            ymin = self.snp[indx[0],1]
            ymax = self.snp[indx[0],1]
            for i in indx:
                if self.snp[i,0] <= xmin:
                    xmin = self.snp[i,0]
                if self.snp[i,0] >= xmax:
                    xmax = self.snp[i,0]
                if self.snp[i,1] <= ymin:
                    ymin = self.snp[i,1]
                if self.snp[i,1] >= ymax:
                    ymax = self.snp[i,1]
            if abs(xmax-xmin) > abs(ymax-ymin):
                side_length = abs(xmax-xmin)
            else:
                side_length = abs(ymax-ymin)
            ax.set_xlim([xmin -.1 ,xmin + side_length +.1])
            ax.set_ylim([ymin -.1 ,ymin + side_length +.1])
            #note, previously, -.6<=x,y<=.6

            voxels = VoxelClick(fig, self.snp, self.sw, self)
            voxels.connect()
            print("-------------------------------------------------")
            print("MaxRod (R,G,B): ",maxangs)
            print("MinRod (R,G,B): ",minangs)
            """write line here for adding text next to the plot"""
            """
            maxs = ','.join(str(np.round_(x,decimals=4)) for x in maxangs)
            mins = ','.join(str(np.round_(x,decimals=4)) for x in minangs)
            plt.figtext(0.76, 0.5, "mins :"+maxs+"\nmaxes:"+mins)
            #plt.tight_layout()
            plt.gcf().subplots_adjust(right=0.75) #adjusting window for text to fit
            """
            plt.show()
            #return voxels.clicked_angles

def simple_plot(snp,sw,plotType,minConfidence,maxConfidence):
    '''
    just plot the location, without orientation information
    :param snp:
    :param sw:
    :return:
    '''
    snp = snp[minConfidence<snp[:,9]<maxConfidence,:]
    plt.plot(snp[:,0],snp[:,1],'*-')
    plt.show()

################# test session ###################
def test_for_dist():
    point = np.array([0.2,0.2])
    line = np.array([[0,0.24],[0.22,0.13]])
    dist = dist_to_line(point,line)
    print('dist should be',dist)
    plt.plot(point[0],point[1])
    plt.plot(line[:,0],line[:,1])
    plt.show()

def test_euler2mat():
    pass
def test_plot_mic():
    sw,snp = read_mic_file('395z0.mic.LBFS')
    #snp = snp[:100,:]
    plot_mic(snp,sw,3,0.35)

def combine_mic():
    sw_82,snp_82 = read_mic_file('Cu_.mic.opt')
    sw_81,snp_81 = read_mic_file('Cu_.mic_opt_81')
    sw_77, snp_77 = read_mic_file('Cu_.mic_opt_77')
    sw_89, snp_89 = read_mic_file('Cu_.mic.opt_89')
    snp = np.concatenate((snp_81,snp_82,snp_77,snp_89), axis=0)
    plot_mic(snp,sw_77,3,0)
    save_mic_file('eulerangles',snp[:,6:9],1)

def test_plot_square_mic():
    sMic = np.load('Au_Mar17_100_100_0.002.npy')
    plot_square_mic(sMic, 0.5)

def test_plot():
    m = MicFile('Au_SYF_.mic.LBFS')
    m.plot_mic_patches(1,0.5)
#if __name__ == '__main__':

    # sw,snp = read_mic_file('1000micron9GenSquare0.5.mic')
    #simple_plot(snp,sw,0,0.5)

    # new_snp = select_line_mic(snp)
    # plt.plot(new_snp[:,0],new_snp[:,1],'*')
    # plt.show()
    # save_mic_file('Cu_line.mic', new_snp, sw)
    #save_mic_file('Cu_combine.mic',snp,sw_82)
    #test_for_dist()
    #test_euler2mat()
    #test_plot_mic()
    #combine_mic()

#clicked_angles = MicFile("395z0.mic.LBFS").plot_mic_patches(1,0.8,1,False,[])
#MicFile("Al_final_z1_refit.mic").plot_mic_patches()

def is_float(x):
    """Function that checks if a string x is that of a numerical value

    Parameters
    ----------
    x : string
        string to be checked if it's a number

    Returns
    -------
    boolean
        True if float, False if not

    """
    try:
        float(x)
    except ValueError:
        return False
    return True

def run():
    """
    Function that runs the main loop

    Returns
    -------


    """
    square_s = input("Is your data file a square matrix file? [y/n]: ")
    assert(square_s == "y" or square_s == "Y" or square_s == "n" or square_s == "N"), "Please enter in 'y' or 'n' format."
    if square_s == "y" or square_s == "Y":
        is_square = True
    else:
        is_square = False

    file_name = input("Please enter file name: ")

    conf_s = input("Please enter minimum confidence threshold: ")
    conf_f = float(conf_s)
    assert(is_float(conf_s)), "Please enter a valid numerical value."

    print("Please select plot type number")
    print("1. Orientation")
    print("2. Confidence levels")
    plottype_s = input("[1/2]: ")
    assert(plottype_s == "1" or plottype_s == "2"), "Please enter either '1' or '2'."
    if plottype_s == "1":
        is_orient = True
    else:
        is_orient = False

    if is_square == True:
        sqm = SquareMic()
        sqm.load(file_name)
        if is_orient:
            misorien_s = input("Please enter misorientation threshold (default=1.0): ")
            try:
                assert(is_float(misorien_s))
                misorien_f = float(misorien_s)
                print(misorien_f)
            except:
                print("Input not a numerical value. Defaulting to 1.0")
                misorien_f = 1.0
            #assert(is_float(misorien_s)), "Please enter a numerical value."
            #misorien_f = float(misorien_s)

            sqm.plot_orientation([],minHitRatio=conf_f,misor_thresh=misorien_f)
        else:
            sqm.plot_hit_ratio()
    else:
        m = MicFile(file_name)
        if is_orient:
            plottype_i = 1
        else:
            plottype_i = 2
        m.plot_mic_patches(plottype_i,conf_f,1,False,[])
    return
