import utils.fflow as flw
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='name of configuration file;', type=str, default='config/pems-bay_config.yml')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # read options
    option = flw.read_option(args.config)
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server
    server = flw.initialize(option)
    # start federated optimization
    server.run()


if __name__ == '__main__':
    main()
