import utils.fflow as flw


def main():
    config_path = 'config/pems-bay_config.yml'
    # read options
    option = flw.read_option(config_path)
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server
    server = flw.initialize(option)
    # start federated optimization
    server.run()


if __name__ == '__main__':
    main()
