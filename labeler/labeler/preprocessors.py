import torch
import numpy as np
from .torchvggish.torchvggish.vggish import VGGish as VGGish_model
import openl3


def get_model(model_name):
    if model_name == "vggish":
        return VGGish()
    elif 'openl3' in model_name:
        params = model_name.split('-')
        model = OpenL3(
            input_repr=params[1],
            embedding_size=int(params[2]), 
            content_type=params[3], 
        )
        return model
    else:
        raise ValueError("couldn't find that model")


class OpenL3:
    def __init__(self, input_repr, embedding_size, content_type):
        import openl3 
        self.model = openl3.models.load_audio_embedding_model(
            input_repr=input_repr,
            embedding_size=embedding_size,
            content_type=content_type
        )

    def __call__(self, x, sr):
        assert isinstance(x, np.ndarray), "input needs to be a numpy array"
        assert isinstance(sr, int), "input needs to be an int"

        assert x.ndim == 1, "input needs to be shape (Frame,). No channel dimension"
        import openl3

        embedding, ts = openl3.get_audio_embedding(x, sr, model=self.model,  verbose=False,
                                             content_type="music", embedding_size=512,
                                             center=False, hop_size=1)

        assert embedding.ndim == 2
        return embedding, ts

class VGGish:
    def __init__(self):
        model_urls = {
            'vggish': 'https://github.com/harritaylor/torchvggish/'
                    'releases/download/v0.1/vggish-10086976.pth',
            'pca': 'https://github.com/harritaylor/torchvggish/'
                'releases/download/v0.1/vggish_pca_params-970ea276.pth'
        }
        self.model = VGGish_model(
                        urls=model_urls, 
                        pretrained=True, 
                        preprocess=True, 
                        postprocess=False, 
                        progress=True)

        # self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.model.eval()

    def __call__(self, x, sr):
        assert isinstance(x, np.ndarray), "input needs to be a numpy array"
        assert isinstance(sr, int), "sr needs to be an int"

        assert x.ndim == 1, "input needs to be shape (Frame,) (no channel dimension)"

        embedding = self.model(x, sr)
        ts = np.array(range(len(embedding))) # luckily, the hop size is 1 

        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().numpy()

        assert embedding.ndim == 2
        return embedding, ts