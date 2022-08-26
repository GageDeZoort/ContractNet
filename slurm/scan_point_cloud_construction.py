import os
import sys
import argparse
import itertools
sys.path.append('../')
sys.path.append('/scratch/gpfs2/jdezoort/oversmoothing/DeepGCNs/')

import numpy as np
import torch_geometric

from gnn_tracking.preprocessing.point_cloud_builder import PointCloudBuilder

# configure initial params
idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
n_evts = 25
n_sectors = [8, 16, 32, 64, 128, 256]
sector_di = [-0.0005, -0.00025, -0.0001, 0, 0.0001, 0.00025, 0.0005]
sector_ds = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
pt_thld = [0.1, 0.5, 0.9]
params = list(itertools.product(n_sectors, sector_di, sector_ds, pt_thld))
p = params[idx]
print(f'n_sectors={p[0]}, sector_di={p[1]}, sector_ds={p[2]}, pt_thld={p[3]}')

# build test graphs
n_sectors = p[0]
sector_di = p[1]
sector_ds = p[2]
pt_thld = p[3]
basedir = '/scratch/gpfs/jdezoort/gnn_tracking/'
pc_builder = PointCloudBuilder(indir=os.path.join(basedir, 'events'),
                               outdir='../../../studies/ContractNet/temp',
                               write_output=False,
                               n_sectors=n_sectors,
                               sector_di=sector_di, 
                               sector_ds=sector_ds, 
                               thld=pt_thld,
                               pixel_only=True, measurement_mode=True,
                               remove_noise=False)
pc_builder.process(n=25, verbose=True)

# extract output measurements
measurements = pc_builder.get_measurements()
print(measurements)
n_hits = measurements["n_hits"]
n_hits_err = measurements["n_hits_err"]
n_hits_ext = measurements["n_hits_ext"]
n_hits_ext_err = measurements["n_hits_ext_err"]
n_hits_ratio = measurements["n_hits_ratio"]
n_hits_ratio_err = measurements["n_hits_ratio_err"]
n_unique_pids = measurements["n_unique_pids"]
n_unique_pids_err = measurements["n_unique_pids_err"]
majority_contained = measurements["majority_contained"]
majority_contained_err = measurements["majority_contained_err"]

# write to output
outfile = os.path.join(basedir, 'studies/ContractNet/slurm/scan_point_cloud_construction.csv')
print('writing to outfile', outfile)
with open(outfile, 'a') as f:
    f.write(f'{n_sectors},{sector_di},{sector_ds},{pt_thld},' +
            f'{n_hits},{n_hits_err},{n_hits_ext},{n_hits_ext_err},' +
            f'{n_hits_ratio},{n_hits_ratio_err},' +
            f'{n_unique_pids},{n_unique_pids_err},' +
            f'{majority_contained},{majority_contained_err}\n')
