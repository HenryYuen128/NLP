class EarlyStopping:
    def __init__(self, metrics_name, mode='min', patience=5):
        self.metrics_name = metrics_name
        self.patience = patience

        self.best_metrics = float("inf") if mode=='min' else float("-inf")
        self.best_all_metrics = None
        self.best_epoch = 0
        self.mode = mode

    def update_best_metrics(self, epoch, **kwargs):
        is_update = False
        if self.mode == 'min':
            if kwargs.get(self.metrics_name) < self.best_metrics:
                self.best_metrics, self.best_epoch = kwargs.get(self.metrics_name), epoch
                self.best_all_metrics = kwargs
                is_update = True
        else:
            if kwargs.get(self.metrics_name) > self.best_metrics:
                self.best_metrics, self.best_epoch = kwargs.get(self.metrics_name), epoch
                self.best_all_metrics = kwargs
                is_update = True
        return is_update

    def is_stop(self, epoch):
        if epoch - self.best_epoch > self.patience:
            return True
        return False