import hydra
from omegaconf import DictConfig, OmegaConf
from conf.config import TrainConfig


from dataset_utils import get_data


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: TrainConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # load dataset
    train_data, test_data = get_data()

    # load model

    # train_model


if __name__ == '__main__':
    main()

