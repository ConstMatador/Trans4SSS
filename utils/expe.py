from torch import nn


class Experiment:
    def __init__(self, conf: Configuration):
        self.conf = conf
        self.device = self.conf.getEntry("device")
        self.epoch_max = self.conf.getEntry("epoch_max")
        self.log_path = conf.getEntry("log_path")
        self.model_path = self.conf.getEntry("model_path")
        
        logging.basicConfig(
            level = logging.INFO,
            format = '%(asctime)s - %(levelname)s - %(message)s',
            filename = self.log_path,
            filemode = "w"
        )
        
        logging.info(f"Experiment initialized with max epochs: {self.epoch_max} on device: {self.device}")