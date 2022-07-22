import os
import sys
sys.path.append('../')
from random import shuffle
from collections import Counter

import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.cluster import DBSCAN

from utils.data_utils import GraphDataset
from models.track_condensation_network import TCN
from models.condensation_loss import *

def div(a,b):
    if b==0: return 0
    return a/b

job_name = 'scan_train1_ptmin0p0'
model_indir = '../hyperparameter_scans/scan_train1_ptmin0p0'
model = os.path.join(model_indir, 'job_12_epoch14.pt')

device = 'cpu'
gnn = TCN(5,4,2,predict_track_params=True).to(device)
gnn.load_state_dict(torch.load(model))

n_sectors = 36
graph_indir = '../graphs/train1_ptmin0p0'
graph_files = np.array(os.listdir(graph_indir))
graph_paths = np.array([os.path.join(graph_indir, f)
                        for f in graph_files])

eps_per_sector = {}
for s in range(n_sectors):
    print(f" --> Sector {s}")
    graphs = [f for f in graph_paths
              if int(f.split('_s')[-1].split('.')[0])==s][0:100]

    params = {'batch_size': 1, 'shuffle': True, 'num_workers': 4}
    test_set = GraphDataset(graph_files=np.array(graphs))
    test_loader = DataLoader(test_set, **params)

    eps_best, eff_best = -0.1, 0.0
    for eps in np.arange(0.1, 1.4, 0.05):
        print(f"...testing eps={eps}")
        tracking_effs = []
        with torch.no_grad():
            for batch, (data, f) in enumerate(test_loader):
                if len(data.x)==0: continue
                out, xc, beta, p = gnn(data.x, data.edge_index, data.edge_attr)
                y, out = data.y, out.squeeze(1)
                particle_id = data.particle_id
                if torch.sum(torch.isnan(data.x)): print('x has nan')
                if torch.sum(torch.isnan(data.edge_index)): print('edge_index has nan')
                if torch.sum(torch.isnan(data.edge_attr)): print('edge_attr has nan')
                if torch.sum(torch.isnan(data.y)): print('y has nan')
                if torch.sum(torch.isnan(data.particle_id)):
                    print(particle_id)
                    print('particle_id has nan')

                n_pids = len(torch.unique(particle_id))
                nhits_per_pid = {}
                has_noise = False
                for pid in torch.unique(particle_id):
                    if pid==0: has_noise = True
                    nhits = torch.sum(particle_id==pid).item()
                    nhits_per_pid[pid.item()] = nhits

                if torch.sum(torch.isnan(xc)):
                    print('skipping nan')
                    continue

                db = DBSCAN(eps=eps, min_samples=2).fit(xc)
                labels = db.labels_
                clusters = np.unique(labels)
                n_clusters = len(clusters)
                double_majority = 0
                for c in clusters:
                    if c==-1: continue
                    xc_cluster = xc[labels==c]
                    x_cluster = data.x[labels==c].numpy()
                    pid_cluster = particle_id[labels==c].tolist()
                    N = len(pid_cluster)
                    pid_counts = Counter(pid_cluster)
                    most_common = pid_counts.most_common(1)[0]
                    N_truth = nhits_per_pid[most_common[0]]
                    if (most_common[1]/N >=0.5) and (most_common[1]/N_truth > 0.5):
                        double_majority+=1

                denom = (n_pids-1) if has_noise else n_pids
                tracking_effs.append(div(double_majority,denom))

        tracking_eff = np.nanmean(tracking_effs)
        if (tracking_eff > eff_best):
            eps_best, eff_best = eps, tracking_eff

    eps_per_sector[s] = eps_best
    print(f"Sector {s}: best eps={eps_best} w/ tracking eff={eff_best}")


