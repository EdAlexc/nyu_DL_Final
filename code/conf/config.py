from dataclasses import dataclass, field
from typing import List, Any, Dict
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf



@dataclass
class LSTMConfig:
    name: str = 'lstm'
    hidden_dim: int = 512
    n_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.2


@dataclass
class DenseConfig:
    name: str = 'dense'
    n_layers: int = 2
    dropout: float = 0.2 

@dataclass
class TransformerConfig:
    name: str = 'transformer'
    dropout: float = 0.2
    nhead: int = 8
    num_layers: int = 2



@dataclass
class TrainConfig:

    defaults: List[Any] = field(default_factory=lambda: [
            {'model': 'lstm'}
        ])

    model = MISSING
    #limit_data: int = None
    limit_data: int = -1
    batch_size: int = 32



cs = ConfigStore.instance()
cs.store(name='train_config', node=TrainConfig)
cs.store(group='model', name='lstm', node=LSTMConfig)
cs.store(group='model', name='dense', node=DenseConfig)
cs.store(group='model', name='transformer', node=TransformerConfig)



