import os
from os.path import join as join
import numpy as np
import pandas as pd
import torch

class GraphBuilder:
    def __init__(self, indir, outdir, pixel_only=True, redo=True,
                 phi_slope_max=0.005, z0_max=200, 
                 dR_max=1.7, uv_approach_max=0.0015,
                 feature_names=["r", "phi", "z", "eta_rz", "u", "v", "layer"],
                 feature_scale=np.array([1000.0, np.pi, 1000.0, 1, 1 / 1000.0, 1 / 1000.0])):
        self.indir=indir
        self.outdir=outdir
        self.pixel_only = pixel_only
        self.redo=redo
        self.phi_slope_max=phi_slope_max
        self.z0_max=z0_max
        self.dR_max=dR_max
        self.uv_approach_max=uv_approach_max
        self.feature_names=feature_names
        self.feature_scale=feature_scale
        self.data_list=[]
        self.outfiles = os.listdir(outdir)
        
    def calc_dphi(self, phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
        """Computes phi2-phi1 given in range [-pi,pi]"""
        dphi = phi2 - phi1
        dphi[dphi > np.pi] -= 2 * np.pi
        dphi[dphi < -np.pi] += 2 * np.pi
        return dphi

    def calc_eta(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Computes pseudorapidity
        (https://en.wikipedia.org/wiki/Pseudorapidity)
        """
        theta = np.arctan2(r, z)
        return -1.0 * np.log(np.tan(theta / 2.0))
    
    def get_dataframe(self, evt, evtid):
        to_df = {'evtid': evtid}
        for i, n in enumerate(self.feature_names):
            to_df[n] = evt.x[:, i]
        to_df['pt'] = evt.pt
        to_df['particle_id'] = evt.particle_id
        return pd.DataFrame(to_df)
    
    def select_edges(self, hits1, hits2, layer1, layer2):
        hit_pairs = (
            hits1.reset_index().merge(hits2.reset_index(), on="evtid", suffixes=("_1", "_2"))
        )
        
        # define various geometric quantities
        dphi = self.calc_dphi(hit_pairs.phi_1, hit_pairs.phi_2)
        dz = hit_pairs.z_2 - hit_pairs.z_1
        dr = hit_pairs.r_2 - hit_pairs.r_1
        eta_1 = self.calc_eta(hit_pairs.r_1, hit_pairs.z_1)
        eta_2 = self.calc_eta(hit_pairs.r_2, hit_pairs.z_2)
        deta = eta_2 - eta_1
        dR = np.sqrt(deta**2 + dphi**2)
        du = hit_pairs.u_2 - hit_pairs.u_1
        dv = hit_pairs.v_2 - hit_pairs.v_1
        m = dv / du
        b = hit_pairs.v_1 - m * hit_pairs.u_1
        u_approach = -m * b / (1 + m**2)
        v_approach = u_approach * m + b
        
        # restrict the distance of closest approach in the uv plane
        uv_approach = np.sqrt(u_approach**2 + v_approach**2)
        
        # restrict phi_slope and z0
        phi_slope = dphi / dr
        z0 = hit_pairs.z_1 - hit_pairs.r_1 * dz / dr

        # apply the intersecting line cut
        intersected_layer = dr.abs() < -1
        # 0th barrel layer to left EC or right EC
        if (layer1 == 0) and (layer2 == 11 or layer2 == 4):
            z_coord = 71.56298065185547 * dz / dr + z0
            intersected_layer = np.logical_and(z_coord > -490.975, z_coord < 490.975)
        # 1st barrel layer to the left EC or right EC
        if (layer1 == 1) and (layer2 == 11 or layer2 == 4):
            z_coord = 115.37811279296875 * dz / dr + z0
            intersected_layer = np.logical_and(z_coord > -490.975, z_coord < 490.975)
            
        # filter edges according to selection criteria
        good_edge_mask = (
            (phi_slope.abs() < self.phi_slope_max)
            & (z0.abs() < self.z0_max)  # geometric
            & (dR < self.dR_max)
            & (uv_approach < self.uv_approach_max)
            & (intersected_layer == False)
        )
        
        # store edges (in COO format) and geometric edge features
        selected_edges = pd.DataFrame(
            {
                "index_1": hit_pairs["index_1"][good_edge_mask],
                "index_2": hit_pairs["index_2"][good_edge_mask],
                "dr": dr[good_edge_mask],
                "dphi": dphi[good_edge_mask],
                "dz": dz[good_edge_mask],
                "dR": dR[good_edge_mask],
            }
        )

        return selected_edges
        
    def build_edges(self, evt):
        if self.pixel_only:
            layer_pairs = [(0,1), (1,2), (2,3), # barrel-barrel
                           (0, 4), (1, 4), (2, 4), (3, 4),  # barrel-LEC
                           (0, 11), (1, 11), (2, 11), (3, 11),  # barrel-REC
                           (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), # LEC-LEC
                           (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17) # REC-REC
                          ]
        else:
            layer_pairs = []
        groups = evt.groupby("layer")
        edges = []
        for (layer1, layer2) in layer_pairs:
            try:
                hits1 = groups.get_group(layer1)
                hits2 = groups.get_group(layer2)
            except KeyError as e:
                continue
            
            edges_layer_pair = self.select_edges(
                hits1,
                hits2,
                layer1,
                layer2,
            )
            edges.append(edges_layer_pair)
        edges = pd.concat(edges)
        edge_attr = np.stack(
            (
                edges.dr.values / self.feature_scale[0],
                edges.dphi.values / self.feature_scale[1],
                edges.dz.values / self.feature_scale[2],
                edges.dR.values,
            )
        )
        node_idx = np.arange(len(evt['r']))
        node_idx = pd.Series(node_idx, index=node_idx)
        edge_start = node_idx.loc[edges.index_1].values
        edge_end = node_idx.loc[edges.index_2].values
        edge_index = np.stack((edge_start, edge_end))
        return edge_index, edge_attr
        
    def process(self, n=10**6, verbose=False):
        infiles = os.listdir(self.indir)
        for f in infiles:
            name = f.split('/')[-1]
            if f in self.outfiles and not self.redo:
                graph = torch.load(join(self.outdir, name))
                self.data_list.append(graph)
            else:
                evtid = f.split('.')[0][5:]
                if verbose: print(f'Processing {f}')
                f = join(self.indir, f)
                graph = torch.load(f)
                df = self.get_dataframe(graph, evtid)
                edge_index, edge_attr = self.build_edges(df)
                graph.edge_index = edge_index
                graph.edge_attr = edge_attr
                if verbose: print(graph)
                outfile = join(self.outdir, name)
                if verbose: print(f'Writing {outfile}')
                torch.save(graph, outfile)
                self.data_list.append(graph)
            
