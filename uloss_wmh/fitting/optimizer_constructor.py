from torch.optim.lr_scheduler import PolynomialLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, ChainedScheduler, LambdaLR, MultiplicativeLR
from torch.optim import Adam, AdamW, SGD

class OptimizerConfigurator():
    def __init__(self, optim, lr, weight_decay, scheduler, nesterov=None, momentum=None, verbose_lr=True, total_epochs=None, min_lr=None, poly_power=None, factor=None, patience=None, threshold=None, monitor=None, cosine_restart_init_iters=None):

        ### setup optimizer
        if optim == 'SGD':
            self.optimizer_constructor = SGD
            self.optim_params = {"lr": lr, "weight_decay":weight_decay, "nesterov":nesterov, "momentum":momentum}
        elif optim == 'Adam':
            if weight_decay > 0:
                self.optimizer_constructor = AdamW # better implementation of weight decay
            else:
                self.optimizer_constructor = Adam
            self.optim_params = {"lr": lr, "weight_decay":weight_decay}
        else:
            raise ValueError("optim should be one of Adam | SGD")


        ### setup learning rate scheduler
        self.monitor=None
        if scheduler == 'Polynomial':
            #print("polynomial scheduler")
            self.lr_schedule_constructor = PolynomialLR
            self.scheduler_params = {"total_iters":total_epochs, "power":poly_power, "verbose":verbose_lr}
        elif scheduler == 'CosineWithRestarts':
            #print("cosine scheduler")
            self.lr_schedule_constructor = CosineAnnealingWarmRestarts
            self.scheduler_params = {"T_0":cosine_restart_init_iters, "T_mult":1, "eta_min":min_lr, "verbose":verbose_lr}
        elif scheduler =='ReduceOnPlateau':
            #print("reduce on plateau scheduler")
            assert monitor != None
            self.monitor = monitor
            #print(monitor)
            self.lr_schedule_constructor = ReduceLROnPlateau
            self.scheduler_params = {"factor":factor, "patience":patience, "threshold":threshold, "min_lr":min_lr, "verbose":verbose_lr}
        elif scheduler == 'Warmup':
            self.lr_schedule_constructor = lambda optimizer : ChainedScheduler([PolynomialLR(optimizer, power=poly_power, total_iters=total_epochs, verbose=verbose_lr), MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.21 if epoch <= 50 else 1 , verbose=True)
            ])
            self.scheduler_params = {}
            assert monitor != None
            self.monitor = monitor
        elif scheduler == 'None' or scheduler == None:
            #print("no lr scheduler")
            self.lr_schedule_constructor = None
        else:
            raise ValueError("scheduler should be one of Polynomial | CosineWithRestarts | ReduceOnPlateau | None")
        

    def __call__(self, parameters):
        optimizer = self.optimizer_constructor(parameters, **self.optim_params)

        if not self.lr_schedule_constructor:
            return optimizer

            
        lr_scheduler = self.lr_schedule_constructor(optimizer, **self.scheduler_params)
        configuration =  {
            "optimizer":optimizer,
            "lr_scheduler": {
                "scheduler":lr_scheduler,
                # 'epoch' updates the scheduler on epoch end, 'step'
                # updates it after a optimizer update.
                "interval":"epoch",
                # rate after every epoch/step.
                "frequency":1,
                
            }
        }

        if self.monitor != None:
            configuration['lr_scheduler']['monitor'] = self.monitor
            
        print(configuration)

        return configuration


def standard_configurations(config_name, total_epochs=1000, lr=0.01):
    configs = {
        "nnunet": OptimizerConfigurator(optim='SGD', lr=0.01, weight_decay=0, nesterov=True, momentum=0.99, scheduler='Polynomial', total_epochs=total_epochs, poly_power=0.9, verbose_lr=True),
        "oddesey": OptimizerConfigurator(optim='Adam', lr=lr, weight_decay=0, scheduler='ReduceOnPlateau', patience=50, threshold=1e-4, factor=0.2, min_lr=1e-6, monitor='train_loss', verbose_lr=True),
        "oddesey_on_val": OptimizerConfigurator(optim='Adam', lr=lr, weight_decay=0, scheduler='ReduceOnPlateau', patience=50, threshold=1e-4, factor=0.2, min_lr=1e-6, monitor='val_loss', verbose_lr=True),
        "challenge": OptimizerConfigurator(optim='Adam', lr=0.0002, weight_decay=0, scheduler='None', verbose_lr=True),
    }

    return configs[config_name]