The ```point_cloud_builder.py``` is abstracted so that you could slot in your own dataset class.
Your builder would inherit from ```BasePointCloudBuilder``` and would need to implement a ```read_event``` and ```process_event``` method. The ```process_event``` will be called by the function ```process``` in a loop over the events. The ```process_event``` method should call your ```read_event method```. In order not to conflict with code downstream, consider using the ```to_pyg_data``` function to save your data in the same format as is expected. Also consider adding tests for your data processing to the appropriate testing script.


Get overview of slum job exit status with

```bash
pipx run reportseff -g --slurm-format build-point-clouds-%A-%a.log
```
