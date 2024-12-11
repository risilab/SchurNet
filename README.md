# SchurNet
Official codebase for SchurNet
# SchurNet

# Package dependencies
1. Install ``ptens`` dev4 branch: https://github.com/risi-kondor/ptens/tree/dev4
2. Install remaining packages listed in requirements.txt

Current config structure:

To create your own experiment, under config/ folder, add an yaml file named experiments.yaml with the following content:
```
defaults:
  - experiment_settings: <file_name>
  - dataset: <file_name>
  - training_hyperparameters: <file_name>
  - model: <file_name>
```

We have uploaded the configs for the experiments in the paper, which includes the ablation studies on TU Dataset, experiments on ZINC, ogbg-molhiv, ogbg-moltox. 

# Training models in single GPU

To train a model in a specific config file, please refer to the following command:
```
    python main_batch.py --config_name=<config_file_name>
```
To debug hydra (in case the configs are incorrectly configured):
```
    HYDRA_FULL_ERROR=1 python main_batch.py --config_name=<config_file_name>
```

To add a new model, create the model under model/ folder. Then include in the config/model folder both the model name, file path, and input parameters needed to initialize the model. 

# Training models in multiple GPUs
To train a model in a specfic config file using multiple gpu, please refer to the following command:
```
./launch_multi_gpu.sh <config_file_name>
```


## TODO:
1. compatible with `dev5` branch of `ptens` package
2. checkpoints for different experiments


