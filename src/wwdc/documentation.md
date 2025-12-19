# WWDC workflow outline

## Making experiments

The experiment should have a folder structure like the following:

experiment
-> meta.yaml
-> sweeps.yaml
-> model.yaml
-> train.yaml
-> embed.yaml

<b>meta.yaml</b> contains all the meta data required to run an experiment. This information is especially important
for tracking outputs from training, embeddings and other stages of the workflow. A typical meta.yaml file looks like the following:

```yaml
meta:
  authors:
  - BR
  catalog_id: 0
  catalog_name: rgdigit
  data_id: 0
  data_name: COLOUR_MNIST
  description: FLOW training on RGB MNIST latent space.
  experiment_id: rgbmnist_1.0.1
  experiment_name: FLOW
  experiment_path: ${paths.data_dir}/${meta.data_name}/${meta.experiment_name}/
  version: 0.1.0
```

<b>sweeps.yaml</b> contains all the information associated with each sweep. This can be structured accordingly for use in train. For example:

```yaml
sweeps:
  sweep_1:
    param: 'x_ds.drop_variables'
    val: '["b", "angle", "scale"]'
    description: 'Drop b, scale and angle'
  sweep_2:
    ...
```

The <b>model.yaml</b> contains a number of key entries:

```yaml
lightning_loader:
  _target_: wwdc.training.lightning_loaders.LightningFlow
  vae:
    _target_: wwdc.models.rgbmnist.VAE
    hidden_dim: 64
  flow:
    _target_: wwdc.models.rgbmnist.Flow
    features: 64
    context: 12
    hidden_features: [64,64]
    transforms: 2
  lr: 0.0005
  size: 64
  batch_size: ${data.loader.batch_size}
  beta_init: 1.0
  beta_floor: 1.0
  vae_ckpt_path: ${paths.vae_ckpt_dir}

model_checkpoint: # this can be more general. Import from checkpoints and change monitor.
  _target_: lightning.pytorch.callbacks.ModelCheckpoint
  dirpath: ${paths.experiment_path}/ckpts/
  save_top_k: 1
  filename: ${hydra:job.id}
  monitor: val_loss
  mode: min

trainer:
  logger:
    - ${logger.wandb}
    - ${logger.csv}

  callbacks:
    - ${...model_checkpoint}

trainer_ckpt_path: null # resume training from this checkpoint if avaialble
```

<b>train.yaml</b> can then use the defined yaml files to construct a training script:

```yaml
# @package _global_

defaults:
- /data/rgbmnist
- /trainer/gpu
- /logger/wandb
- /logger/csv
- /paths/default
- /hydra/default
- model
- sweeps
- meta
- _self_

paths:
  vae_ckpt_dir: ${oc.env:DATA_ROOT}/${meta.data_name}/VAE/ckpts/5239514.ckpt

hydra:
  sweeper:
    params:
      +data.x_ds.drop_variables: ${....sweeps.sweep_1.val}, ${....sweeps.sweep_2.val}, ${....sweeps.sweep_3.val}, ${....sweeps.sweep_4.val}
      +seed: 42, 43, 44
```

Training for the experiment can be ran using the following command:

```python
cd src/wwdc/training
srun python train.py -cn "experiment/RGBMNIST_FLOW/train" hydra/launcher=arc
```

The <b>embedding.yaml</b> file configures the embedding script:

```yaml
# @package _global_

defaults:
  - /data/rgbmnist
  - /trainer/gpu
  - /logger/wandb
  - /paths/default
  - /hydra/default
  - model
  - sweeps
  - meta
  - train
  - _self_


lightning_loader:
  vae_ckpt_path: ${train.paths.vae_ckpt_path}

splits:
  - test

hydra:
  sweeper:
    params:
      +data.x_ds.drop_variables: ${....sweeps.sweep_1.val}, ${....sweeps.sweep_2.val}, ${....sweeps.sweep_3.val}, ${....sweeps.sweep_4.val}
      +seed: 42, 43, 44
```

Run the following to compute the embeddings:

```python
cd src/wwdc/inference
srun python inference.py -cn "experiment/RGBMNIST_FLOW/embed" hydra/launcher=arc
```

## Products

In the ``$DATA/{data_name}/{experiment_name}`` folder the following structure is created after running the train and embed scripts:

$DATA/{data_name}/{experiment_name}
-> ckpts # model checkpoints
-> embeddings # embeddings
-> metrics # results from training runs
-> multiruns # output from hydra/slurm

Each checkpoint and embedding should be accompanied by an info file that contains the sweep data used to create it, for example (52407352_info.yaml):


```yaml
param: x_ds.drop_variables
val: '["scale", "angle", "g"]'
description: Drop green, scale and angle
job_id: 52407352
job_num: 2
vae_fp: /data/dtce-schmidt/phys2526/COLOUR_MNIST/VAE/ckpts/5239514.ckpt
model_fp: /data/dtce-schmidt/phys2526/COLOUR_MNIST/FLOW/ckpts/5240735_2.ckpt
```

## Workflow

- Data Prep (Catalog defintion and dataset)
|
- Train/Test (Model, loss and optimizer defined.)
|
- Embed
|
-- Stat Test (Define tests)
-- Anomaly Detection (Define AD algorithms + Active Learning)
-- Visualisation (Define plots)