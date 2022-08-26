import os
import sys
import argparse
import itertools
sys.path.append('../')
sys.path.append('/scratch/gpfs2/jdezoort/oversmoothing/DeepGCNs/')

import numpy as np
import torch_geometric

from gnn_tracking.graph_construction.graph_builder import GraphBuilder

# configure initial params
idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
n_evts = 25
phi_slope_max = [0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007]
z0_max = [100, 150, 200, 250, 300, 350, 400, 450]
dR_max = [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]
params = list(itertools.product(phi_slope_max, z0_max, dR_max))
p = params[idx]
print(f'phi_slope_max={p[0]}, z0_max={p[1]}, dR_max={p[2]}')

# build test graphs
phi_slope_max = p[0]
z0_max = p[1]
dR_max = p[2]
basedir = '/scratch/gpfs/jdezoort/gnn_tracking/'
graph_builder = GraphBuilder(indir=os.path.join(basedir, 'point_clouds'),
                             outdir='../../../studies/ContractNet/temp',
                             pixel_only=True,
                             redo=True,
                             phi_slope_max=phi_slope_max,
                             z0_max=z0_max,
                             dR_max=dR_max,
                             directed=False,
                             measurement_mode=True,
                             write_output=False)
graph_builder.process(n=n_evts, verbose=True)

# extract output measurements
measurements = graph_builder.get_measurements()
print(measurements)
n_edges = measurements['n_edges']
n_edges_err = measurements['n_edges_err']
n_true_edges = measurements['n_true_edges']
n_true_edges_err = measurements['n_true_edges_err']
n_false_edges = measurements['n_false_edges']
n_false_edges_err = measurements['n_false_edges_err']
n_truth_edge_0 = measurements['n_truth_edge_0']
n_truth_edge_0_err = measurements['n_truth_edge_0_err']
n_truth_edge_0p5 = measurements['n_truth_edge_0.5']
n_truth_edge_0p5_err = measurements['n_truth_edge_0.5_err']
n_truth_edge_0p9 = measurements['n_truth_edge_0.9']
n_truth_edge_0p9_err = measurements['n_truth_edge_0.9_err']
edge_purity = measurements['edge_purity']
edge_purity_err = measurements['edge_purity_err']
edge_efficiency_0 = measurements['edge_efficiency_0']
edge_efficiency_0_err = measurements['edge_efficiency_0_err']
edge_efficiency_0p5 = measurements['edge_efficiency_0.5']
edge_efficiency_0p5_err = measurements['edge_efficiency_0.5_err']
edge_efficiency_0p9 = measurements['edge_efficiency_0.9']
edge_efficiency_0p9_err = measurements['edge_efficiency_0.9_err']

# write to output
outfile = os.path.join(basedir, 'studies/ContractNet/slurm/scan_graph_construction.csv')
print('writing to outfile', outfile)
with open(outfile, 'a') as f:
    f.write(f'{phi_slope_max},{z0_max},{dR_max},' +
            f'{n_edges},{n_edges_err},{n_true_edges},{n_true_edges_err},' +
            f'{n_truth_edge_0},{n_truth_edge_0_err},' +
            f'{n_truth_edge_0p5},{n_truth_edge_0p5_err},' +
            f'{n_truth_edge_0p9},{n_truth_edge_0p9_err},' +
            f'{edge_purity},{edge_purity_err},' +
            f'{edge_efficiency_0},{edge_efficiency_0_err},' +
            f'{edge_efficiency_0p5},{edge_efficiency_0p5_err},' +
            f'{edge_efficiency_0p9},{edge_efficiency_0p9_err}\n')
