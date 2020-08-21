import numpy as np
import torch
from joblib import load

def get_model(model_name, model_kwargs=None):
    if model_name == "jack":
        model = Jack()
    elif model_name == "custom":
        assert 'path_to_model' in model_kwargs, "didn't specify path_to_model in model_kwargs"
        model = torch.load(model_kwargs['path_to_model'])
        model.eval()
    elif model_name == 'openl3_svm-linear_19-class':
        model = LinearSVM(
            path_to_model=model_kwargs['path_to_model'],
            classes=model_kwargs['classes']
        )
    else:
        raise ValueError("incorrect classifier name")

    return model

class LinearSVM:
    def __init__(self, path_to_model, classes):
        self.classes = tuple(classes)
        self.model = load(path_to_model)
        self.pca = load(path_to_model + '_pca')
        # self.fischer = load(path_to_model + '_fischer')

    def predict(self, x, ts=None):
        assert isinstance(x, np.ndarray), "input must be numpy array"
        assert x.ndim == 2, "input must be openl3 embedding with shape (F, 512)"

        print(x.shape)
        x = self.pca.transform(x)

        pred = self.model.predict(x)
        labels = [self.classes[int(i)] for i in pred]
        print(ts.shape)
        ts = list(range(ts[0], ts[-1]+2, 1*(ts[-1]-ts[-2]))) if len(ts) > 1 else [0, 1]
        return labels, ts


class Jack:
    def __init__(self):
        """
        this is the classifier implemented in Jack's AudacityLabeling
        
        the classifier takes in a VGGish embedding
        with batch size 8 and shape (1, 128, 8)
        """
        self.model = torch.load('../mac/Resources/classifier.pt').float()
        self.model.eval()

        self.labels = ('Drums', 'Guitar', 'Strings (continued)', 'Silence')


    def __call__(self, x, ):
        """
        predict class probabilities from input embedding x. 
        x must be a torch tensor of shape (1, 128, 8) 
        """
        assert isinstance(x, torch.Tensor), "input must be torch tensor"
        assert x.size() == (1, 128, 8), "input must have shape (1, 128, 8)"

        return self.model(x)


    def predict(self, x, ts=None):
        """
        predict class labels from input embedding x. 
        x must be a numpy array with shape (F, 128), where F is the number of frames to evaluate

        returns: 
        """
        assert isinstance(x, np.ndarray), "input must be numpy array"
        assert x.ndim == 2 and x.shape[-1] == 128, "input must be vggish embedding with shape (F, 128)"

        x = torch.tensor(x)
        # because this classifier processes in batches of 8, we must drop the last frames if we can't group into 8
        # reshape and drop the last frames if necessary

        # if we can't group everything into groups of 8, we must append some zeros

        if not (len(x) % 8 == 0):
            append_dim = 8 - len(x) % 8
            x = torch.cat((x, torch.zeros((append_dim, 128))), dim=0)

        # now, reshape into batches of 8
        x = x.view(-1, 8, 128)

        preds = []
        for batch in x:
            batch = torch.transpose(batch, 1, 0) # flip so shape is (128, 8)
            batch = batch.unsqueeze(0) # model needs a double tensor and shape (1, 128, 8)
            pred = self(batch).detach().numpy()
            preds.append(pred)

        # since we're using 8 embedding frames to produce one output, we must make a shorter time step
        labels = [self.labels[np.argmax(p)] for p in preds]
        # start at 0, finish at the end of the ts array, step is 8 times bigger
        ts = list(range(ts[0], ts[-1]+1, 8*(ts[1]-ts[0]))) 

        return labels, ts

    

         

    