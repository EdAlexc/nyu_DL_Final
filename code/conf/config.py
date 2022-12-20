from dataclasses import dataclass, field
from typing import List, Any
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf


@dataclass
class LSTMConfig:
    sample: str = 'lstm'
    pass

@dataclass
class DenseConfig:
    sample: str = 'dense'
    pass

@dataclass
class TransformerConfig:
    sample: str = 'transformer'
    pass


@dataclass
class TrainConfig:

    defaults: List[Any] = field(default_factory=lambda: [
            {'model': 'lstm'}
        ])

    model = MISSING
    #limit_data: int = None
    limit_data = None



cs = ConfigStore.instance()
cs.store(name='train_config', node=TrainConfig)
cs.store(group='model', name='lstm', node=LSTMConfig)
cs.store(group='model', name='dense', node=DenseConfig)
cs.store(group='model', name='transformer', node=TransformerConfig)



