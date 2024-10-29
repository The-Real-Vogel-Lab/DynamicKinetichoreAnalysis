# DynamicKinetichoreAnalysis

A pycharm project to read intensity profiles from a fiji macro and determine the separation of the two signals for each frame in a 3D time series video. 

The following is ment to be run as a PyCharm Project. 

The included macro will create an intensity profile across a line selected for each frame. It will save the intensity profile as a `csv` file in the same directory as the image. Each cell treated in this way can be analyzed using the included python script. To do so, complete the following steps: 

1. For each cell you wish to analyze, create a `json` file with the following format: 
    ```{json}
    {
      "cell_id": "EXAMPLE ID",
      "spindle_pole_track_path": "./path/to/trackmate_data.csv",
      "centromere_track_path": "./path/to/trackmate_data.csv",
      "image_path": "./path/to/image_data.czi",
      "strain": "YV3670"
    }
    ```
    The `cell_id` is a unique identifier for each cell to tie the data back to the cell that it came from, the `spindle_pole_track_path` and `centromere_track_path` are the (relative) paths to the trackmate `csv` output for the spindle pole bodies and centromeres for the cells, and the `strain` is an identifier for the strain (mutant or wild type) for the cell.

2. Fill out the `database_processing_settings.env` file. This should have the parameters for where to read the input, save the output, as well as the channel data and the voxel size for converting pixels into distances.

3. Run the python script. It will create a csv for each cell in the database with the following format.
   ```{csv}
    ,Spindle Length,Centromere Separation (Gaussian),Centromere Separation (Trackmate)
    1,1.220808183841571,0.5473600842535923,0.7317078671730208
    2,1.259865798672962,0.5457148695348528,0.7525151200490839
    3,1.2100203442819808,0.4646997210453634,0.5812473719204613
    4,1.2139425835542632,0.08609883879970659,0.5332948182523785
    5,1.3226324831618899,0.8977657843976814,0.5134101322768215
    6,1.0311020533898012,0.6218077941954356,0.5472317016128293
    7,1.15748621009878,0.21539749872492042,0.576363064818607
    8,1.019450289166009,0.44231547947175853,0.5852356944214369
   ```

