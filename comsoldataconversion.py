import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def get_name(path):
    name, _ = os.path.splitext(os.path.basename(path))
    return name


def load_3d(file_name, use_col):
    parameters = np.loadtxt(file_name, delimiter=None, skiprows=9, usecols=(0, 1, 2, use_col),
                            dtype={'names': ('x', 'y', 'z', 'data'),
                                   'formats': ('S4', 'f4', 'f4', 'f4')})
    x_axis = []
    y_axis = []
    z_axis = []
    data = []

    x_points = []
    y_points = []
    z_points = []
    pointnet = []

    for z in range(64):
        for y in range(64):
            for x in range(64):
                x_axis.append(x)
                y_axis.append(y)
                z_axis.append(z)
                index = x + 64 * y + 64 * 64 * z
                if np.isnan(parameters['data'][index]):
                    data.append(0)
                else:
                    data.append(1)
                    x_points.append(x)
                    y_points.append(y)
                    z_points.append(z)

    vox = np.stack((x_axis, y_axis, z_axis, data), axis=1)
    pointnet = np.stack((x_points, y_points, z_points), axis=1)
    # print(vox.shape)
    return vox, pointnet


def load_swept_data(file_name, parameter_number):
    for i in range(parameter_number):
        use_col = i + 3
        vox, pointnet = load_3d(file_name, use_col)
        vox_cluster.append(vox)
        pointnet_cluster.append(pointnet)
        output_name = "3D_output" + str(i) + ".txt"
        np.savetxt(output_name, pointnet, fmt='%2d', newline='\r\n')
        print "3D_output", i, ".txt"
        
    return vox_cluster, pointnet_cluster


vox_cluster = []
pointnet_cluster = []
vox_cluster, pointnet_cluster = load_swept_data("phys-net-v0.3_deform.txt", 201)
print("NOW DEFORM")

