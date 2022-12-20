import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from conf.config import TrainConfig


from dataset_utils import get_data
from models import ModelCatalog
from train_utils import train


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: TrainConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # load dataset
    train_data, test_data, labels = get_data(limit_data = cfg.limit_data, batch_size = cfg.batch_size)
    #train_data, test_data = buildDataLoader(train_data, cfg.batch_size), buildDataLoader(test_data, cfg.batch_size)

    # load model
    model = ModelCatalog[cfg.model.name](output_dim=len(labels), **cfg.model)

    
    # construct path for storing logs from run
    model_path = []
    for k, v in cfg.model.items():
        model_path.extend([str(k), str(v)])

    # train_model
    log_dir = Path(
       '/scratch/rd2893/dl_runs' ,
       *model_path
    )
    results = train(model, train_data, test_data, 100, log_dir)
    # save results
    with open(Path(log_dir, 'results.json'), 'w+') as f:
        f.write(json.dumps(results))


    



if __name__ == '__main__':
    main()

