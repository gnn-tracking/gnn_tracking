# GNNs for Charged Particle Tracking

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/GageDeZoort/gnn_tracking/main.svg)](https://results.pre-commit.ci/latest/github/GageDeZoort/gnn_tracking/main)
[![gh actions](https://github.com/GageDeZoort/gnn_tracking/actions/workflows/test.yaml/badge.svg)](https://github.com/GageDeZoort/gnn_tracking/actions)


## Development setup

Install the pre-commit hooks with

```bash
pip3 install pre-commit
# cd to this directory
pre-commit install
```

## Organization (to-do)
- /src/gnn_tracking
  - preprocessing
    - build_point_clouds
    - get_particle_properties
  - graph_construction
    - build_graphs
    - build_module_map
  - utils
    - graph_datasets
      - initialize_logger()
      - get_graph_paths()
      - get_graph_evtids()
      - sort_graph_paths()
      - clean_graph_paths()
      - partition_graphs()
      - get_graph_dataset()
      - get_dataloader()
      - GraphDataset()
    - segmentation
    - preprocessing
      - open_yaml()
      - map_pt()
      - get_trackml_prefixes()
      - load_module_map()
      - filter_file_prefixes()
      - relabel_pids()
    - graph_construction
      - initialize_logger()
      - calc_dphi()
      - calc_eta()
      - empty_graph()
      - split_detector_sectors()
      - select_hits()
      - select_edges()
      - correct_truth_labels()
      - get_particle_properties()
      - get_n_track_segs()
      - graph_summary()
    - training
    - plotting
      - plot_rz()
      - plot_3d()
    - batch_jobs
    - losses
      - EdgeWeightLoss()
      - PotentialLoss()
      - BackgroundLoss()
      - ObjectLoss()
  - models
    - IN
    - TCN1
    - TCN2
    - EC
  - training
    - ECTrainer
    - TCN1Trainer
    - TCN2Trainer
  - postprocessing
    - GraphClustering
    - measure_tracking_effs
    - TuneDBSCAN
- slurm
- notebooks
- configs
- examples
- tests
