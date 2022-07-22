import os
import sys
sys.path.append('../')

import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits import mplot3d
import mplhep as hep
plt.style.use(hep.style.CMS)

def plot_rz(X, idxs, y, save_fig=False, filename="rz.png"):
    X = np.array(X)
    feats_o = X[idxs[0,:]]
    feats_i = X[idxs[1,:]]
    
    fig = plt.figure(dpi=200)
    for i in range(len(X)):        
        plt.scatter(X[i][2], X[i][0], c='silver', linewidths=0, marker='s', s=15)

    track_segs_o = feats_o[y>0.5]
    track_segs_i = feats_i[y>0.5]
    for i in range(len(track_segs_o)):
        plt.plot((track_segs_o[i][2], track_segs_i[i][2]),
                 (track_segs_o[i][0], track_segs_i[i][0]),
                  marker='o', ls='-', color='blue', lw=0.25, ms=0.1, alpha=1)

    false_edges_o = feats_o[y<0.5]
    false_edges_i = feats_i[y<0.5]
    for i in range(len(false_edges_o)):
        plt.plot((false_edges_o[i][2], false_edges_i[i][2]),
                 (false_edges_o[i][0], false_edges_i[i][0]),
                  marker='o', ls='-', color='black', lw=0.15, ms=0.1, alpha=0.4)
        
    plt.ylabel("r [m]")
    plt.xlabel("z [m]")
    #plt.title(f'Sector: ({label[0]}, {label[1]})')
    if (save_fig): plt.savefig(filename, dpi=1200)
    plt.tight_layout()
    plt.show()
    
def plot_3d(X, idxs, y, save_fig=False, filename="rz.png"):
    X = np.array(X)
    r, phi, z = X[:,0], X[:,1], X[:,2]
    pred = y
    x, y = r*np.cos(phi), r*np.sin(phi)
    x_o, y_o, z_o = x[idxs[0,:]], y[idxs[0,:]], z[idxs[0,:]]
    x_i, y_i, z_i = x[idxs[1,:]], y[idxs[1,:]], z[idxs[1,:]]
    
    #feats_o = X[idxs[0,:]]
    #feats_i = X[idxs[1,:]]
    
    fig = plt.figure(figsize=(12,12), dpi=200)
    ax = plt.axes(projection='3d')
    for i in range(len(X)):  
        ax.scatter3D(x, y, z, c='silver', marker='s', s=15);
        #plt.scatter(X[i][2], X[i][0], c='silver', linewidths=0, marker='s', s=8)

    xt_o, yt_o, zt_o = x_o[pred>0.5], y_o[pred>0.5], z_o[pred>0.5]
    xt_i, yt_i, zt_i = x_i[pred>0.5], y_i[pred>0.5], z_i[pred>0.5]
    #track_segs_o = feats_o[y>0.5]
    #track_segs_i = feats_i[y>0.5]
    for i in range(len(xt_o)):
        ax.plot3D((xt_o[i], xt_i[i]), (yt_o[i], yt_i[i]), (zt_o[i], zt_i[i]),
                   marker='o', ls='-', color='blue', lw=0.25, ms=0, alpha=1)

    xf_o, yf_o, zf_o = x_o[pred<0.5], y_o[pred<0.5], z_o[pred<0.5]
    xf_i, yf_i, zf_i = x_i[pred<0.5], y_i[pred<0.5], z_i[pred<0.5]
    #false_edges_o = feats_o[y<0.5]
    #false_edges_i = feats_i[y<0.5]
    for i in range(len(xf_o)):
        ax.plot3D((xf_o[i], xf_i[i]), (yf_o[i], yf_i[i]), (zf_o[i], zf_i[i]),
                   marker='h', ls='-', color='black', lw=0.15, ms=0, alpha=0.4)
        
    ax.set_xlabel("x [m]", labelpad=25)
    ax.set_ylabel("y [m]", labelpad=25)
    ax.set_zlabel("z [m]", labelpad=25)
    #plt.title(f'Sector: ({label[0]}, {label[1]})')
    #if (save_fig): plt.savefig(filename, dpi=1200)
    plt.show()
