import copy
import torch

class EMA:
    def __init__(self, model, decay=0.995):
        self.ema_model = copy.deepcopy(model).eval()
        self.decay = decay

        for p in self.ema_model.parameters():
            p.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
                ema_p.data = self.decay * ema_p.data + (1 - self.decay) * p.data