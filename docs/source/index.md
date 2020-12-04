# README

## Project overview

The project is splitted into two parts

- [Train](./src.train): This module contains the code to train a model
- [Runtime](./src.runtime): Contains the code for everything besides training, eg testing, demo or production usage

There's also some shared code between these two parts which is located under [common](./src.common).

The [scripts](scripts) namespace contains many scripts "around" the project, eg to convert/prepare Datasets or to record
testing data from [CARLA](https://carla.org/).

## Datasets

The dataset has to meet our dataset definition exactly. If the dataset doesn't fulfill the specification exactly it will
fail. See [creating a dataset](./howto/create_a_profile.md) for details.

## Usage

The main application is used by running the `ufld.py` file. The usage of the scripts is different per script. See their documentation / the howtos for details on their usage.

### Setup

See [HOWTO: Setup](./howto/setup.md)

### Config/CLI Parameters

Every parameter has a default value, specified in the default profile. Every config entry can be overridden in a custom
config and also via CLI. To use config options via the CLI interface prefix them with `--`, eg `--batch_size 16`. All
available options are documented under `ufld.py --help` and [configs](./configs).

An example command for training could look like

``` shell
python ufld.py configs/carla_trainset02.py --mode train
```

To test your model you could use a command like

``` 
python ufld.py configs/carla_trainset02.py --mode test
```

For further examples see [HOWTOs](./howto/howtos.md)

## Collecting data from CARLA

CARLA is one way to get training and test data. Sadly CARLA doesn't provide the required information over its API. As a
result it's quite complicated to generate a dataset from CARLA. We wrote some scripts which make this possible. The
scripts are located in the [scripts](./scripts) folder. Also have a
look [on this HOWTO](howto/generate_dataset_from_carla.md) which explains their usage.

## HOWTOs

The [HOWTOs](./howto/howtos.md) section provides guides for every common usecase/problem of this application.



```{toctree}
---
maxdepth: 1
caption: Contents
---
howto/howtos
modules
```


   