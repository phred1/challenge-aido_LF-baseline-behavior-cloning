# Dataset collection
To generate our dataset, we used the logs from this file for the real images:
[FH_real_ds_1.log] https://udemontreal-my.sharepoint.com/:u:/g/personal/anthony_courchesne_umontreal_ca/EXN3k7Z0VshBu4XZQaEeVOoBQdkjhHXAS8p8BGpQ3ZREmw?e=0fqkLO

Credits to Francois Hebert for making the tedious work of collecting those logs!

For the Real images logs, we used this log file:

[ds_300_150slim_ir_yuv.log](https://udemontreal-my.sharepoint.com/:u:/g/personal/anthony_courchesne_umontreal_ca/EU4kZ4VktRVLsssJ2I81Ka8BAscRdbizPJuIvqG0HDks3w?e=dadOmW)

Credits to Anthony Courchesne for the logs. 

From those logs, to collect the dataset, simply run the following commands at the root of the repo:
```
cd duckieLog/
mkdir dataset_raw/real
mkdir dataset_raw/sim
```
Before generating the dataset, modify the variable `ENV` in the 
`util/log_viewer.py`
```
ENV="real" or ENV="sim"
```

Then for the simulator dataset
```
python3 util/log_viewer.py --log_name ds_300_150slim_ir_yuv.log 
```
and for the real dataset:
```
python3 util/log_viewer.py --log_name FH_real_ds_1.log
```

You should now have two folders with 38751 elements each. This is the dataset we used for our project. 

## UNIT
The UNIt dataset is tricky to generate since it requires the files to respect a specific folder structure and needs a train/test split to work. 

This is the desired folder structure:

 ```
    /dataset/sim2real_raw
            - test
                - images_a:
                    - 0001.jpeg
                    - 0002.jpeg
                    - 0003.jpeg
                - images_b
                    - 0001.jpeg
                    - 0002.jpeg
                    - 0003.jpeg
            - train
                - images_a
                    - 0001.jpeg
                    - 0002.jpeg
                    - 0003.jpeg
                - images_b
                    - 0001.jpeg
                    - 0002.jpeg
                    - 0003.jpeg
 ```

To facilitate the generation of such dataset, the file `train_test_splitter.py` enables a few transformations.

To generate a dataset in the UNIT format, simply run

```
python3 util/train_test_splitter.py  
```

This will produce an 80/10 train-test split of the dataset and produce the desired folder structure.


