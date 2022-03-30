from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule, Trainer
import math

from pytorch_lightning.callbacks import GradientAccumulationScheduler, StochasticWeightAveraging


class batch_size_schedule():

    def __init__(
        self,
        args,
        total_epoch: int = 1,
    ):
        """
            Args: -> passing the total epochs training

        """

        self.args = args
        self.total_epoch = total_epoch

    def batch_increase(self):
        # till 30 % epoch, it will run original Global Batch_size.
        # From 31% epoch till 70% epoch it will run  Global Batch_size*2
        # From 70% till end it will run  Global Batch_size*3

        stage_1 = 30

        stage_2 = int(self.total_epoch * 0.3 + 1)

        stage_3 = int(self.total_epoch*0.7)

        accumulator = GradientAccumulationScheduler(
            scheduling={stage_1: 1, stage_2: 2, stage_3: 3})

        return accumulator

    def batch_increase_decrease(self):
        # till 30 % epoch, it will run original Global Batch_size*2.
        # From 31% epoch till 70% epoch it will run  Global Batch_size
        # From 70% till end it will run  Global Batch_size*3

        stage_1 = 30

        stage_2 = int(self.total_epoch * 0.3 + 1)

        stage_3 = int(self.total_epoch*0.7)

        accumulator = GradientAccumulationScheduler(
            scheduling={stage_1: 2, stage_2: 1, stage_3: 3})

        return accumulator


class Stochastic_Weight_Avg():
    def __init__(
        self,
        args,
        swa_start_epochs: float = 0.5,
    ):
        """
            Args: -> passing the swa_start_epochs starting SWA

        """
        self.args = args
        self.swa_start_epochs = swa_start_epochs

    def SWA_on_epochs(self):
        ##the procedure will start from the swa_epoch_start-th epoch
        swa = StochasticWeightAveraging(swa_epoch_start=self.swa_start_epochs)
        return swa


class Alpha_schedule(Callback):
    """
        1. Alpha schedule with Proportion of training epochs
        2. Cosine Schedule.
        Note:: Automatically increases (Alpha and Beta) from ``initial_value`` to 1.0 with  training step

    """

    def __init__(
        self,
        args,
        total_epoch: int = 1,
        init_alpha: float = 0.5,
        alpha="schedule"
    ):
        """
            Args: -> for cosine schedule
                initial_value: starting value. Auto-updates with every training step
        """
        super().__init__()
        self.args = args
        self.total_epoch = total_epoch
        if "cosine_schedule" in alpha:
            try:
                self.init_alpha = float(alpha.split("_")[-1])
            except Exception as e:
                print("cosine schedule formate error")
                self.init_alpha = init_alpha
        else:
            self.init_alpha = init_alpha

        self.current_alpha = self.init_alpha
        self.alpha = alpha



    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> float:

        if self.alpha == "schedule":
            # print("You implement alpha schedule")
            epoch = trainer.current_epoch
            if epoch < self.total_epoch * 0.3:
                pl_module.alpha = 0.5
            elif epoch < self.total_epoch * 0.9:
                pl_module.alpha = 0.8
            else:
                pl_module.alpha = 1

        elif "cosine_schedule" in self.alpha:
            # print("You implement Alpha schedule")
            max_steps = len(trainer.train_dataloader) * trainer.max_epochs
            self.current_alpha = 1 - \
                (1 - self.init_alpha) * (math.cos(math.pi *
                                                  pl_module.global_step / max_steps) + 1) / 2
            pl_module.alpha = self.current_alpha
        else:
            pl_module.alpha = float(self.alpha)


class Beta_schedule(Callback):
    """
        1. Beta schedule with Proportion of training epochs
        2. Beta Schedule.
        Note:: Automatically increases (Alpha and Beta) from ``initial_value`` to 1.0 with  training step

    """

    def __init__(
        self,
        args,
        total_epoch: int = 1,
        init_beta: float = 0.5,
        beta="schedule"
    ):
        """
            Args: -> for cosine schedule
                initial_value: starting value. Auto-updates with every training step
        """
        super().__init__()
        self.args = args
        self.total_epoch = total_epoch
        if "cosine_schedule" in beta:
            try:
                self.init_beta = float(beta.split("_")[-1])
            except Exception as e:
                print("cosine schedule formate error")
                self.init_beta = init_beta
        else:
            self.init_beta = init_beta

        self.beta = beta
        self.current_beta = self.init_beta

    def on_train_epoch_start(self, trainer: Trainer, module: LightningModule):
        if self.beta == "schedule":
            epoch = trainer.current_epoch
            if epoch < self.total_epoch * 0.5:
                module.beta = 0.6
            elif epoch < self.total_epoch * 0.7:
                module.beta = 0.4
            else:
                module.beta = 0.2

        elif "cosine_schedule" in self.beta :
            # print("You implement Beta schedule")
            max_steps = len(trainer.train_dataloader) * trainer.max_epochs
            # self.current_beta = 1 - (1 - self.init_beta) * (math.cos(math.pi * module.global_step / max_steps) + 1) / 2
            self.current_beta = self.init_beta * (math.cos(math.pi * module.global_step / max_steps) + 1) / 2

            module.beta = self.current_beta
        else:
            module.beta = float(self.beta)
