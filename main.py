from lightning.pytorch.cli import LightningCLI
from data.module import DataModule
from models.module import CLS


def cli_main():
    cli = LightningCLI(CLS, DataModule,parser_kwargs={"parser_mode": "omegaconf"})
    # note: don't call fit!!


if __name__ == '__main__':
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if-block