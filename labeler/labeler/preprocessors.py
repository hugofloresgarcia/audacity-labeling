import torch
import numpy as np
from .torchvggish.torchvggish.vggish import VGGish as VGGish_model

def get_model(model_name, model_kwargs=None):
    if model_name == "vggish":
        return VGGish()
    elif model_name == "OpenL3":
        if model_kwargs is None:
            model_kwargs = dict(
                input_repr="mel128", 
                embedding_size=512, 
                content_type="music"
            )
        return OpenL3(
            model_kwargs["input_repr"],
            model_kwargs["embedding_size"], 
            model_kwargs["content_type"]
        )
    else:
        raise ValueError("couldn't find that model")

#TODO: rewrite these models as nn.modules to do batch processing and optimize training.
#TODO: though, you can't do batch processing with audio because it's not uniform.
class OpenL3:
    def __init__(self, input_repr, embedding_size, content_type):
        # LAZY LOADING !! tensorflow b heavy
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

        embedding, ts = openl3.get_audio_embedding(x, sr, model=self.model,  verbose=False,
                                             content_type="music", embedding_size=512,
                                             center=True, hop_size=1)

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

        #TODO this may not work if audio files are different sample rates, may need to iterate instead.
        # lets do the mean var, dmean dvar on the vggish embedding
        embedding = self.model(x, sr)
        ts = np.array(range(len(embedding))) # luckily, the hop size is 1 

        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().numpy()

        return embedding, ts