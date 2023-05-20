import torch

class Predicter():
    def __init__(self, model_sam, prompter):
        self.model_sam = model_sam
        self.prompter = prompter

    def predict(self, x):
        self.model_sam.eval()
        with torch.no_grad():
            prompt = self.prompter(x)
            return self.model_sam.predict(x, **prompt)