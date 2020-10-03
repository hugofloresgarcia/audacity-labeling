import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .openl3_torch import OpenL3Mel128

class TunedOpenL3(pl.LightningModule):

    def __init__(self, hparams=None):
        """
        deployment version of the model
        """
        super().__init__()
        self.classes = list("saxophone,flute,guitar,contrabassoon,bass-clarinet,trombone,cello,oboe,bassoon,banjo,mandolin,tuba,viola,french-horn,english-horn,violin,double-bass,trumpet,clarinet".split(','))

        self.openl3 = OpenL3Mel128()
        self.fc_seq = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(), 
            nn.Dropout(p=0),

            nn.BatchNorm1d(128),
            nn.Linear(128, 19)
        )

    def forward(self, x):
        x = self.openl3(x)
        # x = x.view(-1, 512)
        x = self.fc_seq(x)

        return x
    
    def predict(self, x, ts=None, get_probs=False):
        """
        predict class labels from audio

        if get_probs is False, the function returns
            tuple -> (predictions, ts) where
                predictions: string predictions, list[str] with shape (batch,)
        
        if get_probs is True, the function returns
            probits: probability distribution over labels for each frame
                    with shape (batch, classes)

        """
        self.eval()
        self.openl3.eval()
        with torch.no_grad:
            assert x.ndim == "audio must be mono audio with shape (batch, time)"

            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)

            # audio coming in is shape (batch, time)
            # reshape to (batch, channel, time)
            x = x.unsqueeze(1)

            # get a melsepec
            x = self.openl3.melspec(x)

            # get logits
            x = self(spec)
            probits = F.softmax(x, dim=1)

            yhat = torch.argmax(probits, dim=1, keepdim=False)

            predictions = [self.classes[int(idx)] for idx in yhat]
        
            if get_probs:
                return probits

        return predictions, ts