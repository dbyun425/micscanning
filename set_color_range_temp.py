def set_color_range_sq(smdCopy,x,y,indx,mat,quat,rod):
    for i in range(mat.shape[0]):
        yi = i/x
        xi = i%x
        if (xi,yi) in indx:
            quat[i, :] = RotRep.quaternion_from_matrix(mat[i, :, :])
            rod[i, :] = RotRep.rod_from_quaternion(quat[i, :])
            if (xi,yi) == indx[0]:
                maxr = rod[i,0]
                minr = rod[i,0]
                maxg = rod[i,1]
                ming = rod[i,1]
                maxb = rod[i,2]
                minb = rod[i,2]
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
    for j in range(N):
        for k in range(0,3):
            colors[j,k] = (rod[j,k]-minrgb[k])/(maxrgb[k]-minrgb[k])
    return return colors, maxangs, minangs

def set_color_range(mic, N, indx, mat, quat, rod):
    """
    Function for setting the color range of a plot.
    """
    for i in range(N):
        if i in indx:
            mat[i,:,:] = RotRep.EulerZXZ2Mat(mic.snp[i,6:9]/180.0*np.pi)
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
    print("Current rod values: ",rod)
    maxrgb = [maxr,maxg,maxb]
    minrgb = [minr,ming,minb]
    maxangs = [rod[maxri,0],rod[maxgi,1],rod[maxbi,2]]
    minangs = [rod[minri,0],rod[mingi,1],rod[minbi,2]]
    colors = rod
    for j in range(N):
        for k in range(0,3):
            colors[j,k] = (rod[j,k]-minrgb[k])/(maxrgb[k]-minrgb[k])
    return colors, maxangs, minangs
