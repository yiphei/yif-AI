from dataclasses import dataclass


@dataclass
class BatchStatsBase:
    model: dataclass
    train_config: dataclass

    @classmethod
    def initialize(cls, train_config, model):
        return cls(model, train_config)

    def add_mini_batch_stats(self, *args, **kwargs):
        pass

    def mean(self):
        pass

    def scale(self):
        pass

    def get_wandb_batch_stats(self):
        return {}
