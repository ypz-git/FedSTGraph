# FedSTGraph: A Benchmark of Spatio-Temporal Graphs for Federated Settings 🚧

This repo is a benchmark for spatio-temporal graph data under federated (distributed) scenarios. We collect 12 spatio-temporal datasets among different Real-world scenarios.

## Table of Contents

- [Requirements](#Requirements)
- [Quick Start](#Quick-Start)
- [Datasets](#Datasets-📐)
- [Partition Settings](#Partition-Settings)
- [Architecture](#Architecture)

## Requirements

```txt
numpy>=1.17.2
pytorch>=1.3.1
torchvision>=0.4.2
matplotlib>=3.1.1
prettytable>=2.1.0
ujson>=4.0.2
```

## Quick Start

**1. To get a federated partition of METR-LA dataset**
```shell
python generate_fedtask.py --benchmark metr-la_traffic_forecasting
```

**2. To train METR-LA in federated system**
```shell
python main.py --config config/metr-la_config.yml
```

**3. To get a visualization of the result**
```shell
python result_analysis_forecasting.py
```

## Datasets

| IDX | Name             | Type                | Task                        | Node          | Number of nodes | Timespan              | Time granularity | Source                                                                                          | Support          |
|:---:|:----------------:|:-------------------:|:---------------------------:|:-------------:|:---------------:|:---------------------:|:----------------:|:-----------------------------------------------------------------------------------------------:|:----------------:|
|  1  |PEMS-BAY          |Traffic speed        |Traffic forecasting          |Sensors        |325              |01/01/2017 - 31/05/2017|5 min             |[[github]](https://github.com/liyaguang/DCRNN)                                                   |:heavy_check_mark:|
|  2  |METR-LA           |Traffic speed        |Traffic forecasting          |Sensors        |207              |01/03/2012 - 30/06/2012|5 min             |[[github]](https://github.com/liyaguang/DCRNN)                                                   |:heavy_check_mark:|
|  3  |PEMS03            |Traffic speed        |Traffic forecasting          |Sensors        |358              |09/01/2018 - 11/30/2018|5 min             |[[github]](https://github.com/Davidham3/STSGCN)                                                  |:hammer:                  |
|  4  |PEMS04            |Traffic speed        |Traffic forecasting          |Sensors        |307              |01/01/2018 - 02/28/2018|5 min             |[[github]](https://github.com/Davidham3/STSGCN)                                                  |:hammer:                  |
|  5  |PEMS07            |Traffic speed        |Traffic forecasting          |Sensors        |883              |05/01/2017 - 08/31/2017|5 min             |[[github]](https://github.com/Davidham3/STSGCN)                                                  |:hammer:                  |
|  6  |PEMS08            |Traffic speed        |Traffic forecasting          |Sensors        |170              |07/01/2016 - 08/31/2016|5 min             |[[github]](https://github.com/Davidham3/STSGCN)                                                  |:hammer:                  |
|  7  |Shuto-Expy        |Traffic speed        |Traffic forecasting          |Sensors        |1843             |10/01/2021 - 12/31/2021|10 min            |[[github]](https://github.com/deepkashiwa20/MegaCRN)                                             |                  |
|  8  |Traffic           |Traffic speed        |Traffic forecasting          |Sensors        |862              |01/01/2015 - 12/31/2016|1 hour            |[[github]](https://github.com/laiguokun/multivariate-time-series-data)                           |                  |
|  9  |Solar-Energy      |Resource consumption |Solar power forecasting      |PV plants      |137              |01/01/2006 - 12/31/2006|10 min            |[[github]](https://github.com/laiguokun/multivariate-time-series-data)                           |                  |
| 10  |Electricity       |Resource consumption |Electricity usage forecasting|Users          |321              |01/01/2012 - 12/31/2014|1 hour            |[[github]](https://github.com/laiguokun/multivariate-time-series-data)                           |                  |
| 11  |Exchange-Rate     |Exchange rate        |Exchange rate forecasting    |Countries      |8                |01/01/1990 - 12/31/2016|1 day             |[[github]](https://github.com/laiguokun/multivariate-time-series-data)                           |                  |
| 12  |SDWPF             |Resource consumption |Wind power forecasting       |Turbogenerators|134              |245 days               |10 min            |[[Baidu KDD CUP 2022]](https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction)|:hammer:                  |

## Partition Settings

For METR-LA & PEMS-BAY dataset, we partition dataset to each client hold one node(sensor) data.

## Architecture

TODO :hammer:

**P.S. This is basically refer to the older version of [easyFL: A Lightning Framework for Federated Learning](https://github.com/WwZzz/easyFL).**

