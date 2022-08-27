import numpy as np
import argparse
import random
import torch
import os.path
import importlib
import os
import utils.fmodule
import ujson
import time
import collections
import utils.network_simulator as ns
import yaml

sample_list=['uniform', 'md']
agg_list=['uniform', 'weighted_scale', 'weighted_com']
optimizer_list=['SGD', 'Adam']
logger = None


def read_option(config_path):
    with open(config_path, 'r') as f:
        cfg = f.read()
        option = yaml.load(cfg, Loader=yaml.SafeLoader)
    return option


def setup_seed(seed):
    random.seed(1+seed)
    np.random.seed(21+seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(12+seed)
    torch.cuda.manual_seed_all(123+seed)


def initialize(option):
    # init fedtask
    print("init fedtask...", end='')
    # dynamical initializing the configuration with the benchmark
    bmk_name = option['task'][:option['task'].find('cnum')-1].lower()
    bmk_model_path = '.'.join(['benchmark', bmk_name, 'model', option['model']])
    bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])
    utils.fmodule.device = torch.device('cuda:{}'.format(option['gpu']) if torch.cuda.is_available() and option['gpu'] != -1 else 'cpu')
    utils.fmodule.TaskCalculator = getattr(importlib.import_module(bmk_core_path), 'TaskCalculator')
    utils.fmodule.TaskCalculator.setOP(getattr(importlib.import_module('torch.optim'), option['optimizer']))
    # The Model is defined in bmk_model_path as default, whose filename is option['model'] and the classname is 'Model'
    # If an algorithm change the backbone for a task, a modified model should be defined in the path 'algorithm/method_name.py', whose classname is option['model']
    try:
        utils.fmodule.Model = getattr(importlib.import_module(bmk_model_path), 'Model')
    except ModuleNotFoundError:
        utils.fmodule.Model = getattr(importlib.import_module('.'.join(['algorithm', option['algorithm']])), option['model'])
    model = utils.fmodule.Model().to(utils.fmodule.device)
    # init the model that owned by the server (e.g. the model trained in the server-side)
    try:
        utils.fmodule.SvrModel = getattr(importlib.import_module(bmk_model_path), 'SvrModel')
    except:
        utils.fmodule.SvrModel = utils.fmodule.Model
    # init the model that owned by the client (e.g. the personalized model whose type may be different from the global model)
    try:
        utils.fmodule.CltModel = getattr(importlib.import_module(bmk_model_path), 'CltModel')
    except:
        utils.fmodule.CltModel = utils.fmodule.Model
    # load pre-trained model
    try:
        if option['pretrain'] != '':
            model.load_state_dict(torch.load(option['pretrain'])['model'])
    except:
        print("Invalid Model Configuration.")
        exit(1)
    # read federated task by TaskPipe
    TaskPipe = getattr(importlib.import_module(bmk_core_path), 'TaskPipe')
    train_datas, valid_datas, test_data, client_names = TaskPipe.load_task(os.path.join('fedtask', option['task']))
    num_clients = len(client_names)
    print("done")

    # init client
    print('init clients...', end='')
    client_path = '%s.%s' % ('algorithm', option['algorithm'])
    Client=getattr(importlib.import_module(client_path), 'Client')
    clients = [Client(option, name = client_names[cid], train_data = train_datas[cid], valid_data = valid_datas[cid]) for cid in range(num_clients)]
    print('done')

    # init server
    print("init server...", end='')
    server_path = '%s.%s' % ('algorithm', option['algorithm'])
    server = getattr(importlib.import_module(server_path), 'Server')(option, model, clients, test_data = test_data)
    # init virtual network environment
    ns.init_network_environment(server)
    # init logger
    try:
        Logger = getattr(importlib.import_module(server_path), 'MyLogger')
    except AttributeError:
        Logger = DefaultLogger
    global logger
    logger = Logger()
    print('done')
    return server


def output_filename(option, server):
    header = "{}_".format(option["algorithm"])
    for para in server.paras_name: header = header + para + "{}_".format(option[para])
    output_name = header + "M{}_R{}_B{}_E{}_NS{}_LR{:.4f}_P{:.2f}_S{}_LD{:.3f}_WD{:.3f}_DR{:.2f}_AC{:.2f}_.json".format(
        option['model'],
        option['num_rounds'],
        option['batch_size'],
        option['num_epochs'],
        option['num_steps'],
        option['learning_rate'],
        option['proportion'],
        option['seed'],
        option['lr_scheduler']+option['learning_rate_decay'],
        option['weight_decay'],
        option['net_drop'],
        option['net_active'])
    return output_name


class Logger:
    def __init__(self):
        self.output = collections.defaultdict(list)
        self.current_round = -1
        self.temp = "{:<30s}{:.4f}"
        self.time_costs = []
        self.time_buf={}

    def check_if_log(self, round, eval_interval=-1):
        """For evaluating every 'eval_interval' rounds, check whether to log at 'round'."""
        self.current_round = round
        return eval_interval > 0 and (round == 0 or round % eval_interval == 0)

    def time_start(self, key = ''):
        """Create a timestamp of the event 'key' starting"""
        if key not in [k for k in self.time_buf.keys()]:
            self.time_buf[key] = []
        self.time_buf[key].append(time.time())

    def time_end(self, key = ''):
        """Create a timestamp that ends the event 'key' and print the time interval of the event."""
        if key not in [k for k in self.time_buf.keys()]:
            raise RuntimeError("Timer end before start.")
        else:
            self.time_buf[key][-1] =  time.time() - self.time_buf[key][-1]
            print("{:<30s}{:.4f}".format(key+":", self.time_buf[key][-1]) + 's')

    def save(self, filepath):
        """Save the self.output as .json file"""
        if len(self.output)==0: return
        with open(filepath, 'w') as outf:
            ujson.dump(dict(self.output), outf)
            
    def write(self, var_name=None, var_value=None):
        """Add variable 'var_name' and its value var_value to logger"""
        if var_name==None: raise RuntimeError("Missing the name of the variable to be logged.")
        self.output[var_name].append(var_value)
        return

    def log(self, server=None):
        pass


class DefaultLogger(Logger):
    def __init__(self):
        super(DefaultLogger, self).__init__()

    def log(self, server=None, current_round=-1):
        if len(self.output) == 0:
            self.output['meta'] = server.option
        test_metric = server.test()
        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)
        # calculate weighted averaging of metrics of training datasets across clients
        # train_metrics = server.test_on_clients(self.current_round, 'train')
        # for met_name, met_val in train_metrics.items():
        #     self.output['train_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(server.client_vols, met_val)]) / server.data_vol)
        # calculate weighted averaging and other statistics of metrics of validation datasets across clients
        valid_metrics = server.test_on_clients(self.current_round, 'valid')
        for met_name, met_val in valid_metrics.items():
            self.output['valid_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(server.client_vols, met_val)]) / server.data_vol)
            self.output['mean_valid_' + met_name].append(np.mean(met_val))
            self.output['std_valid_' + met_name].append(np.std(met_val))
        # output to stdout
        for key, val in self.output.items():
            if key == 'meta': continue
            print(self.temp.format(key, val[-1]))