import argparse
import yaml
from runner import Runner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('yamlParams', help='Settings & parameters yaml file path.')
    parser.add_argument('-e', default=300, help='Set number of episodes to run', type=int)
    parser.add_argument('-m', default=None, help='Set file for loading a pre-existing model.', type=str)
    parser.add_argument('-l', default='log.txt', help='Set output log file name.', type=str)
    parser.add_argument('-c', default='checkpoint.model', help='Set checkpoint file name.', type=str)
    parser.add_argument('-f', default=1, help='Set checkpoint save frequency.', type=int)
    parser.add_argument('-q', action='store_false', help='Quiet run. Non-verbose printout of episode details.')
    kwargs = vars(parser.parse_args())

    with open(kwargs['yamlParams'], 'r') as r:
        params = yaml.load(r.read(), Loader=yaml.FullLoader)
    runner = Runner(agentParams=params['agentParams'], envParams=params['environmentParams'],
                    logFile=kwargs['l'], checkpointName=kwargs['c'], checkpointFreq=kwargs['f'])
    if kwargs['m'] is not None:
        runner.loadParameters(kwargs['m'])
    runner.run(kwargs['e'], kwargs['q'])


if __name__ == '__main__':
    main()