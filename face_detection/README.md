## Face Detection
This subdir includes face detection related codes. Some descriptions has 
been presented in repo README.md. 

### Recent Update
* `2019.10.14` The model v2 can be tried to train nightly.
* `2019.10.16` **The model v2 can be trained normally.**

### Brief Introduction to Model Version
* v1 - refer to the paper for details
* v2 - the detection scale is 10-320 (vs 10-560 in v1), the number of layers is 20, 
the backbone is modified for faster inference. Refer to `./net_farm/symbol_structures.xlsx` for details.

### Accuracy
on the way

### Inference Latency
on the way

### User Instructions
> **Now only for traning v2 nightly.**  

First, we introduce the functionality of each sub directory.
* [net_farm](net_farm). This folder contains net definitions for all model versions.
* [metric_farm](metric_farm). This folder contains the metrics for training monitoring.
* [data_provider_farm](data_provider_farm). This folder contains the code of raw data processing/formatting/packing&unpacking.
* [data_iterator_farm](data_iterator_farm). This folder contains the code of multi-threaded data prefetching. 
**This is the most important part, since it describe the essence of LFFD!!!**
* [config_farm](config_farm). This folder contains the configurations of all model versions. The training is started by running the corresponding config python script.

Second, we present a common procedure for running the code for training (taking v2 as an example).

1. prepare net model `net_farm/naivenet.py`
2. prepare the training data by using the code in `data_provider_farm`. We provide a packed 
training data of WIDERFACE trainset. Please download from **Data Download**.
3. adjust the code around the line 241 in `data_iterator_farm/multithread_dataiter_for_cross_entropy_v2`.
4. set the variables in configuration py script in `config_farm`.
5. run `python configuration_10_320_20L_5scales_v2.py` in `config_farm` directory.

### Data Download
We have packed the training data of WIDERFACE train set. In the data, the faces less than 8 pixels are ignored, and some pure negative 
images cropped from the training images are also added. We provide three ways to download the packed data:
* [Baidu Yunpan](https://pan.baidu.com/s/1a8Wk4GNkfPYbKAFSrZzFIQ) (pwd:e7bv)
* [MS OneDrive](https://1drv.ms/u/s!Av9h0YMgxdaSgwiP4nKDasu4m73J?e=v5UfWQ)
* [Google Drive](https://drive.google.com/open?id=1O3nJ6mQKD_sdFpfXmYoK7xnTUg3To7kO)

After you download the data, you can put it anywhere. Remember to set `param_trainset_pickle_file_path` variable in the configuration file. (we 
usually put the data into the folder: `./data_provider_farm/data_folder/`)
