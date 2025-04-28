The ```point_cloud_builder.py``` is abstracted so that you could slot in your own dataset class.
Your builder would inherit from ```BasePointCloudBuilder``` and would need to implement a ```read_event``` and ```process_event``` method. The ```process_event``` will be called by the function ```process``` in a loop over the events. The ```process_event``` method should call your ```read_event method```. In order not to conflict with code downstream, consider using the ```to_pyg_data``` function to save your data in the same format as is expected. Also consider adding tests for your data processing to the appropriate testing script.


Get overview of slum job exit status with

```bash
pipx run reportseff -g --slurm-format build-point-clouds-%A-%a.log
```


            edge_index=self._get_edge_index(hits["particle_id"].values),
            y=torch.zeros(0).float(),
            layer=torch.tensor(hits.layer_id.values).long(),
            particle_id=torch.tensor(hits["particle_id"].values).long(),
            pt=torch.tensor(hits["pt"].values).float(),
            reconstructable=torch.tensor(hits["reconstructable"].values).long(),
            sector=torch.tensor(hits["sector"].values).long(),
            eta=torch.tensor(hits["eta_pt"].values).float(),
            n_hits=torch.tensor(hits["n_hits"].values).long(),
            n_layers_hit=torch.tensor(hits["n_layers_hit"].values).long(),
