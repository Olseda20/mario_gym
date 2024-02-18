import os
from stable_baselines3.common.callbacks import BaseCallback

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq=5000 , save_path=CHECKPOINT_DIR, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'best_model_{self.n_calls}')
            self.model.save(model_path)

        return True

# TODO: save the best models over time?

