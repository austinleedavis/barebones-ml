import os

import datasets
import hydra
import pyrootutils
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)
from transformers import Trainer as HfTrainer
from transformers import (
    TrainingArguments,
)

from callbacks import NtfyCallback
from dataset_processor import DatasetProcessor

load_dotenv()

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(root / "configs"),
    "config_name": "train.yaml",
}


class Trainer:

    config: DictConfig
    dataset: datasets.Dataset
    hf_trainer: HfTrainer
    hf_trainer_args: TrainingArguments
    llm: GPT2LMHeadModel
    tokenizer: PreTrainedTokenizerFast

    def __init__(self, config: DictConfig):
        self.config: DictConfig = config
        self.ntfy = NtfyCallback(topic=os.environ["NTFY_TOPIC"])

        self.llm = hydra.utils.call(config.llm.from_pretrained)
        self.tokenizer = hydra.utils.instantiate(config.llm.tokenizer.instance)
        self.tokenizer.model_max_length = self.llm.config.n_ctx

        # setup and process dataset
        self.dataset = hydra.utils.call(config.dataset.load_dataset)
        if config.run.debug:
            self.dataset = self.dataset.select(range(10000))
        transforms = config.dataset.get("transforms", [])
        processor = DatasetProcessor(transforms)
        self.dataset = processor.process(self.dataset)

        self.dataset = self.dataset.map(lambda ex: self.tokenizer(ex["text"]), desc="Tokenize")

        # setup Hf Trainer
        self.hf_trainer_args = hydra.utils.instantiate(config.training)
        collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.hf_trainer = HfTrainer(
            self.llm,
            args=self.hf_trainer_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            data_collator=collator,
            processing_class=self.tokenizer,
            callbacks=[self.ntfy],
        )

    def train(self):
        print("Train!")
        try:
            self.hf_trainer.train()
        finally:
            self.ntfy.send_notification("Training failed")
            if wandb.run:
                wandb.run.finish(exit_code=-1)


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main(None)
