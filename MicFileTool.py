'''
NOTE TO FUTURE SELF: IDEAS!
Automatically take in account +/- 15% angles (or maybe 10)
Then add a slider which goes from 0 to 15 via .01%
To update, just set the alpha color of that specific triangle = 0, or rgb = [0,0,0]
'''
'''
Originally writen by He Liu
Wed Apr 26 2017
This script will contains the basic tool for reading mic file and plot them.

Modifications for coloring made by Doyee Byun
Including References to VoxelTool and Replotting Modifications written by Grayson Frazier
April 10, 2018

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
from VoxelTool import VoxelBorders


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
        print(rod)
        fig, ax = plt.subplots()
        ax.scatter(snp[:,0],snp[:,1],s=scattersize,facecolors=(rod+np.array([1,1,1]))/2)
        ax.axis('scaled')
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
        new_indx = []
        xl = angles[0]- .15*angles[0]
        xh = angles[0]+ .15*angles[0]
        yl = angles[1]- .15*angles[1]
        yh = angles[1]+ .15*angles[1]
        zl = angles[2]- .15*angles[2]
        zh = angles[2]+ .15*angles[2]
        for i in range(len(indx)):
            j = indx[i]
            x = self.snp[j,6]
            y = self.snp[j,7]
            z = self.snp[j,8]
            if x >= xl and x <= xh and y >= yl and y <= yh and z >= zl and z <= zh:
                new_indx.append(indx[i])
        return new_indx

    def plot_mic_patches(self,plotType=1,minConfidence=0,maxConfidence=1,limitang=False,angles=[]):
        indx = []
        not_indx = []
        for i in range(0,len(self.snp)): #limits snp based on confidence
            if self.snp[i,9] >= minConfidence and self.snp[i,9] <= maxConfidence:
                indx.append(i)
            else:
                not_indx.append(i)
        if limitang: #inputs the replot device
            indx = self.angle_limiter(indx,self.snp,angles)
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
                maxr = 0.0
                minr = 0.0
                maxg = 0.0
                ming = 0.0
                maxb = 0.0
                minb = 0.0
                for i in indx:
                    mat[i,:,:] = RotRep.EulerZXZ2Mat(self.snp[i,6:9]/180.0*np.pi)
                    quat[i,:] = RotRep.quaternion_from_matrix(mat[i,:,:])
                    rod[i,:] = RotRep.rod_from_quaternion(quat[i,:])
                    if i == indx[0]:
                        maxr = rod[i,0]
                        minr = rod[i,0]
                        maxg = rod[i,1]
                        ming = rod[i,1]
                        maxb = rod[i,2]
                        minb = rod[i,2]
                    else:
                        if rod[i,0] > maxr:
                            maxr = rod[i,0]+ .01
                        elif rod[i,0] < minr:
                            minr = rod[i,0]-.01
                        if rod[i,1] > maxg:
                            maxg = rod[i,1]+.01
                        elif rod[i,1] < ming:
                            ming = rod[i,1]-.01
                        if rod[i,2] > maxb:
                            maxb = rod[i,2]+.01
                        elif rod[i,2] < minb:
                            minb = rod[i,2]-.01
                for i in not_indx:
                        rod[i,:]=[0.0,0.0,0.0]
                print("Current rod values: ",rod)
                maxrgb = [maxr,maxg,maxb]
                minrgb = [minr,ming,minb]
                colors = rod
                for j in range(N):
                    for k in range(0,3):
                        colors[j,k] = (rod[j,k]-minrgb[k])/(maxrgb[k]-minrgb[k])
                self.color1= colors
                print("Color: ", self.color1)
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
            plt.axis("equal")
            #note, previously, -.6<=x,y<=.6

            data_borders = VoxelBorders(self.snp, self.sw)

            voxels = VoxelClick(fig, self.snp, self.sw, self)
            voxels.connect()
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
MicFile("395z0.mic.LBFS").plot_mic_patches()#2,0.8,1,False,[])
#MicFile("Al_initial_z1_refit.mic").plot_mic_patches()
#MicFile("Al_final_z1_refit.mic").plot_mic_patches()
#MicFile("Al_final_z1_refit.mic").plot_mic_patches()
