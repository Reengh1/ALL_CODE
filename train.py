import argparse
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner
import easy_tpp
import swanlab

def main(dataset:str):
    swanlab.init(
    project="THP",
    experiment_name=f"Easy TPP MLE Amazon Scaled",
    config={
        "model": "THP",
        "dataset": dataset,
        "batch_size": 1,
        "lr": 1e-3,
    }
    )
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_dir', type=str, required=False, default='/home/guangchen_li/dev/DCL_TPP/configs/experiment_config.yaml',
                        help='Dir of configuration yaml to train and evaluate the  model.')

    parser.add_argument('--experiment_id', type=str, required=False, default='THP_train',
                        help='Experiment id in the config file.')

    args = parser.parse_args()

    config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)

    model_runner = Runner.build_from_config(config)

    model_runner.run()


if __name__ == '__main__':
    print(easy_tpp.__file__)
    main("Amazon")