from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
#from pytorch_lightning.callbacks import TQDMProgressBar
import os
import pytorch_lightning as pl

accelerator="gpu"
devices=1
precision = 32

def get_trainer(max_epochs, results_dir, early_stop_patience, use_early_stopping=True, early_stop_on_train=False):
    
    checkpoint_callback = ModelCheckpoint(results_dir, save_top_k=2, monitor="val_loss")

    callbacks = [checkpoint_callback]

    if use_early_stopping:

        if early_stop_on_train:
            early_stop_callback = EarlyStoppingOnTraining(monitor="train_loss", min_delta=0.01, patience=early_stop_patience, verbose="False", mode="min", check_finite=True)
        else:
            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=early_stop_patience, verbose="False", mode="min", check_finite=True)
        callbacks.append(early_stop_callback)
    
    trainer = pl.Trainer(
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        precision=precision,
        default_root_dir=results_dir
    )
    
    return trainer



# class EarlyStoppingOnTraining(EarlyStopping):
#     # use this to early stop on training when there is no validation fold.

#     def on_validation_end(self, trainer, pl_module):
#         # override this to disable early stopping at the end of val loop
#         pass

#     def on_train_end(self, trainer, pl_module):
#         # instead, do it at the end of training loop
#         self._run_early_stopping_check(trainer, pl_module)