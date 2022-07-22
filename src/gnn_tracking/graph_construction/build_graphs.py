import os
import sys
import argparse
import logging
import multiprocessing as mp
from functools import partial
from collections import Counter
from datetime import datetime
from os.path import join
sys.path.append('../')

import yaml
import pickle
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 
import trackml.dataset
import time
from torch_geometric.data import Data

from utils.graph_building_utils import *
from utils.hit_processing_utils import *

def parse_args(args):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/prepare_trackml.yaml')
    add_arg('--n-workers', type=int, default=1)
    add_arg('--task', type=int, default=0)
    add_arg('--n-tasks', type=int, default=1)
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    add_arg('--start-evtid', type=int, default=0)
    add_arg('--end-evtid', type=int, default=100000)
    add_arg('--redo', type=bool, default=True)
    return parser.parse_args(args)

def construct_graph(hits, layer_pairs, 
                    phi_slope_max, z0_max, dR_max, uv_approach_max, 
                    feature_names, feature_scale, evtid="-1",
                    module_map={}, s=(-1,-1), use_module_map=False):
    """ Loops over hits in layer pairs and extends edges
        between them based on geometric and/or data-driven
        constraints. 
    """
    # loop over layer pairs, assign edges between their hits
    groups = hits.groupby('layer')
    edge_df = []
    for (layer1, layer2) in layer_pairs:
        try:
            hits1 = groups.get_group(layer1)
            hits2 = groups.get_group(layer2)
        except KeyError as e: 
            continue
            
        # assign edges based on geometric and data-driven constraints
        selected_edges = select_edges(hits1, hits2, layer1, layer2,
                                      phi_slope_max, z0_max, dR_max,  # geometric 
                                      uv_approach_max=uv_approach_max,
                                      module_map=module_map,          # data-driven
                                      use_module_map=use_module_map)  
        edge_df.append(selected_edges)

    edge_df = pd.concat(edge_df)
    logging.debug(f'Selected {len(edge_df)} edges.')

    # if no edges, return empty graph
    if len(edge_df)==0:
        return empty_graph()
    
    # prepare the graph matrices
    n_nodes = hits.shape[0]
    n_edges = len(edge_df)
    
    # select and scale relevant features
    x = (hits[feature_names].values / feature_scale).astype(np.float32)
    edge_attr = np.stack((edge_df.dr.values/feature_scale[0], 
                          edge_df.dphi.values/feature_scale[1], 
                          edge_df.dz.values/feature_scale[2], 
                          edge_df.dR.values))
    y = np.zeros(n_edges, dtype=np.float32)

    # use a series to map hit label-index onto positional-index.
    node_idx = pd.Series(np.arange(n_nodes), index=hits.index)
    edge_start = node_idx.loc[edge_df.index_1].values
    edge_end = node_idx.loc[edge_df.index_2].values
    edge_index = np.stack((edge_start, edge_end))

    # fill the edge, particle labels
    # true edges have the same pid, ignore noise (pid=0)
    pid1 = hits.particle_id.loc[edge_df.index_1].values
    pid2 = hits.particle_id.loc[edge_df.index_2].values
    y[:] = ((pid1 == pid2) & (pid1>0) & (pid2>0)) 
    y, n_corrected = correct_truth_labels(hits, edge_df[['index_1', 'index_2']], 
                                          y, pid1)
    
    # check if edges are both in overlap region
    id1 = hits.hit_id.loc[edge_df.index_1].values
    id2 = hits.hit_id.loc[edge_df.index_2].values
    edge_hit_id = np.stack((id1, id2), axis=-1)
    
    particle_id = hits['particle_id']
    if np.sum(np.isnan(x)): logging.info('WARNING: x contains nan!')
    if np.sum(np.isnan(particle_id)): logging.info('WARNING: particle_id contains nan!')

    return {'x': x, 'hit_id': hits['hit_id'],
            'particle_id': particle_id,
            'edge_index': edge_index, 'edge_attr': edge_attr, 
            'y': y, 's': s, 'n_corrected': n_corrected,
            'edge_hit_id': np.transpose(edge_hit_id)}

def process_event(prefix, output_dir, module_map, pt_min, 
                  n_eta_sectors, n_phi_sectors,
                  eta_range, phi_range, 
                  phi_slope_max, z0_max, dR_max, uv_approach_max, 
                  endcaps, remove_noise, remove_duplicates,
                  use_module_map, phi_overlap, eta_overlaps,
                  measurement_mode=False, save_graphs=True,
                  base_dir=''):
    
    # define valid layer pair connections 
    layer_pairs = [(0,1), (1,2), (2,3)]                     # barrel-barrel
    if endcaps:
        layer_pairs.extend([(0,4), (1,4), (2,4), (3,4),     # barrel-LEC
                            (0,11), (1,11), (2,11), (3,11), # barrel-REC
                            (4,5), (5,6), (6,7),            # LEC-LEC
                            (7,8), (8,9), (9,10), 
                            (11,12), (12,13), (13,14),      # REC-REC
                            (14,15), (15,16), (16,17)])
                                 
    # load the data
    evtid = int(prefix[-9:])
    logging.info('Event %i, loading data' % evtid)
    hits, particles, truth = trackml.dataset.load_event(
        prefix, parts=['hits', 'particles', 'truth'])

    # apply hit selection
    logging.info('Event %i, selecting hits' % evtid)
    hits, particles = select_hits(hits, truth, particles, pt_min, endcaps, 
                                  remove_noise, remove_duplicates)
    hits = hits.assign(evtid=evtid)

    # get truth information for each particle
    hits_by_particle = hits.groupby('particle_id')
    particle_properties = pd.read_csv(join(base_dir, f'particle_properties/{evtid}.csv'),
                                      header=0)

    # add conformal coordinates
    hits['u'] = hits['x']/(hits['x']**2 + hits['y']**2)
    hits['v'] = hits['y']/(hits['x']**2 + hits['y']**2)
    hits = hits[['hit_id', 'r', 'phi', 'eta', 'u', 'v', 'z', 
                 'evtid', 'layer', 'module_id', 'particle_id']]

    # map non-reconstructable particles to noise
    initial_noise = np.sum(hits.particle_id==0)
    pid_n_layers_hit = particle_properties[['particle_id', 'n_layers_hit']].values
    pid_n_layers_hit = {p: p*(n>1) for p, n in pid_n_layers_hit}
    hits['particle_id'] = hits.particle_id.map(pid_n_layers_hit)
    addtl_noise = np.sum(hits.particle_id==0) - initial_noise
    logging.info(f"Assigned {addtl_noise} non-reconstructable particles as noise.")

    # map other truth quantities to particles
    track_props = particle_properties[['particle_id', 'pt', 'd0', 'q', 
                                        'reconstructable']].values
    track_props = {p[0]: [p[1], p[2], p[3], p[4]] for p in track_props}

    # divide detector into sectors
    phi_edges = np.linspace(*phi_range, num=n_phi_sectors+1)
    eta_edges = np.linspace(*eta_range, num=n_eta_sectors+1)
    hits_sectors, sector_info = split_detector_sectors(hits, phi_edges, eta_edges,
                                                       phi_overlap=phi_overlap,
                                                       eta_overlaps=eta_overlaps)
    logging.debug(f'Sectors divided as {sector_info}')

    # calculate particle truth in each sector
    #n_track_segs_per_s = {}
    #for s, hits_sector in hits_sectors.items():
    #    hits_sector_by_particle = hits_sector.groupby('particle_id')
    #    n_track_segs_s = get_n_track_segs(hits_sector_by_particle, set(layer_pairs))
    #    n_track_segs_per_s[s] = n_track_segs_s
    
    # graph features and scale
    feature_names = ['r', 'phi', 'z', 'u', 'v']
    feature_scale = np.array([1000., np.pi / n_phi_sectors, 1000., 
                              1/1000., 1/1000.])

    # Construct the graph
    logging.info('Event %i, constructing graphs' % evtid)
    sectors = [construct_graph(sector_hits, layer_pairs=layer_pairs,
                               phi_slope_max=phi_slope_max, 
                               z0_max=z0_max, dR_max=dR_max,
                               uv_approach_max=uv_approach_max,
                               s=s, feature_names=feature_names,
                               feature_scale=feature_scale,
                               evtid=evtid, module_map=module_map,
                               use_module_map=use_module_map)
               for s, sector_hits in hits_sectors.items()]
    output = {'sectors': sectors}

    if (measurement_mode):
        logging.info('Event %i, calculating graph summary' % evtid)
        summary_stats = graph_summary(evtid, sectors, particle_properties,
                                      sector_info, print_per_layer=False)
        output['summary_stats'] = summary_stats
    else:
        mask = (particle_properties.pt > pt_min)
        n_track_segs = particle_properties['n_track_segs'].values[mask]
        n_track_segs = np.sum(n_track_segs)
        n_true = np.sum([np.sum(s['y']) for s in sectors])
        n_false = np.sum([np.sum(s['y']==0) for s in sectors])
        output['summary_stats'] = {'efficiency': n_true/n_track_segs,
                                   'purity': n_true/(n_true+n_false)}
        logging.info(f"Output statistics: {output['summary_stats']}")

    # Write these graphs to the output directory
    base_prefix = os.path.basename(prefix)
    filenames = [os.path.join(output_dir, f'{base_prefix}_s{i}')
                 for i in range(len(sectors))]
    
    logging.info('Event %i, writing graphs', evtid)
    if save_graphs:
        for sector, filename in zip(sectors, filenames):
            track_param_scale = np.array([1., 0.01, 1.])
            track_params = np.array([[(track_props[p][0]-1.40183)/(1.86927-1.14929),
                                      (track_props[p][1])/(0.00663+0.00631),
                                      int(track_props[p][2] > 0)]
                                     for p in sector['particle_id']])
            print(track_params)
            reconstructable = np.array([track_props[p][3] for p in sector['particle_id']])
            np.savez(filename, **dict(x=sector['x'], y=sector['y'], 
                                      track_params=track_params,
                                      reconstructable=reconstructable,
                                      particle_id=sector['particle_id'],
                                      edge_attr=sector['edge_attr'],
                                      edge_index=sector['edge_index']))
        
    return output

def main(args):
    """Main function"""
    args = parse_args(args)
    initialize_logger(verbose=args.verbose)
    config = open_yaml(args.config)
    selection = config['selection']
    logging.debug(f'Selection: {selection}')
    
    pt_min = selection['pt_min']
    pt_str = map_pt(pt_min)
    
    input_dir = config['input_dir']
    base_dir = config['base_dir']
    train_idx = int(input_dir.split('train_')[-1][0])
    logging.info(f"Running on train_{train_idx} data")
    
    # load in hit file prefixes within specified evtid range
    evtid_min, evtid_max = args.start_evtid, args.end_evtid
    file_prefixes = get_file_prefixes(input_dir, trackml=True, codalab=True,
                                      evtid_min=evtid_min, evtid_max=evtid_max,
                                      n_tasks=args.n_tasks, task=args.task)
    
    if selection['use_module_map']:
        load_module_map(config['module_map_dir'])
    else: module_map = []

    output_dir = os.path.expandvars(config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f'Writing outputs to {output_dir}')
    if not args.redo:
        file_prefixes = filter_file_prefixes(file_prefixes, output_dir)

    with mp.Pool(processes=args.n_workers) as pool:
        process_func = partial(process_event, output_dir=output_dir,
                               base_dir=base_dir,
                               phi_range=(-np.pi, np.pi), 
                               module_map=module_map,
                               **config['selection'])
        output = pool.map(process_func, file_prefixes)
    
    # analyze output statistics
    logging.info('All done!')
    graph_sectors = [graph['sectors'] for graph in output]
    summary_stats = [graph['summary_stats'] for graph in output]

    n_nodes = np.array([graph_stats['n_nodes'] 
                        for graph_stats in summary_stats])
    n_edges = np.array([graph_stats['n_edges'] 
                        for graph_stats in summary_stats])
    purity = np.array([graph_stats['purity'] 
                       for graph_stats in summary_stats])
    efficiency = np.array([graph_stats['efficiency'] 
                           for graph_stats in summary_stats])
    boundary_fraction = np.array([graph_stats['boundary_fraction'] 
                                  for graph_stats in summary_stats])
    logging.info(f'Events ({evtid_min},{evtid_max}), average stats:\n' +
                 f'...n_nodes: {n_nodes.mean():.0f}+/-{n_nodes.std():.0f}\n' +
                 f'...n_edges: {n_edges.mean():.0f}+/-{n_edges.std():.0f}\n' + 
                 f'...purity: {purity.mean():.5f}+/-{purity.std():.5f}\n' + 
                 f'...efficiency: {efficiency.mean():.5f}+/-{efficiency.std():.5f}\n' + 
                 f'...boundary fraction: {boundary_fraction.mean():.5f}+/-{boundary_fraction.std():.5f}')
    

    # output to csv
    graph_level_stats = pd.DataFrame({'n_nodes': n_nodes, 'n_edges': n_edges,
                                      'purity': purity, 'efficiency': efficiency})
    now = datetime.now()
    dt_string = now.strftime("%m-%d-%Y_%H-%M")
    outfile = 'stats/'+dt_string+f'_train{train_idx}_ptmin{pt_str}.csv'
    logging.info(f'Saving summary stats to {outfile}')

    f = open(outfile, 'w')
    f.write('# args used to build graphs\n')
    for arg in vars(args):
        f.write(f'# {arg}: {getattr(args,arg)}\n')
    for sel, val in config['selection'].items():
        f.write(f'#{sel}: {val}\n')

    graph_level_stats.to_csv(f, index=False)
    
    # analyze per-sector statistics
    sector_stats_list = [graph_stats['sector_stats'] 
                         for graph_stats in summary_stats]
    neta_sectors = config['selection']['n_eta_sectors']
    nphi_sectors = config['selection']['n_phi_sectors']
    num_sectors = neta_sectors*nphi_sectors
    eta_range_per_s = {s: [] for s in range(num_sectors)}
    phi_range_per_s = {s: [] for s in range(num_sectors)}
    n_nodes_per_s = {s: [] for s in range(num_sectors)}
    n_edges_per_s = {s: [] for s in range(num_sectors)}
    purity_per_s = {s: [] for s in range(num_sectors)}
    efficiency_per_s = {s: [] for s in range(num_sectors)}
    for sector_stats in sector_stats_list:
        for s, stats in sector_stats.items():
            eta_range_per_s[s] = stats['eta_range']
            phi_range_per_s[s] = stats['phi_range']
            n_nodes_per_s[s].append(stats['n_nodes'])
            n_edges_per_s[s].append(stats['n_edges'])
            purity_per_s[s].append(stats['purity'])
            efficiency_per_s[s].append(stats['efficiency'])
            
    for s in range(num_sectors):
        eta_range_s = eta_range_per_s[s]
        phi_range_s = phi_range_per_s[s]
        n_nodes_s = np.array(n_nodes_per_s[s])
        n_edges_s = np.array(n_edges_per_s[s])
        purity_s = np.array(purity_per_s[s])
        efficiency_s = np.array(efficiency_per_s[s])
        logging.info(f'Event ({evtid_min},{evtid_max}), Sector {s}, average stats:\n' +
                     f'...eta_range: ({eta_range_s[0]:.3f},{eta_range_s[1]:.3f})\n' + 
                     f'...phi_range: ({phi_range_s[0]:.3f},{phi_range_s[1]:.3f})\n' + 
                     f'...n_nodes: {n_nodes_s.mean():.0f}+/-{n_nodes_s.std():.0f}\n' +
                     f'...n_edges: {n_edges_s.mean():.0f}+/-{n_edges_s.std():.0f}\n' + 
                     f'...purity: {purity_s.mean():.5f}+/-{purity_s.std():.5f}\n' + 
                     f'...efficiency: {efficiency_s.mean():.5f}+/-{efficiency_s.std():.5f}')
        
if __name__ == '__main__':
    main(sys.argv[1:])
