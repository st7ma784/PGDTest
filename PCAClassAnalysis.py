# prompt: find classes in coco dataset and then repeat for other datasets like fgvc and food101
import json
import os
import clip
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
pca = PCA(n_components=2)
class_names = {}
with open(os.path.join(".","train_class_names.json"),'r') as f:
    class_names = json.load(f)

tokens={}
model, preprocess = clip.load("ViT-B/32",device='cpu')
with torch.inference_mode(True):
    for key in class_names.keys():
        names=class_names[key]
        print("datasets: ",key)
        print("names: ",names)
        names=clip.tokenize(names)
        tokens.update({key:model.encode_text(names).detach().cpu().numpy()})

fullpoints=np.cat(tokens.values(),axis=0)
X_pca = pca.fit_transform(fullpoints)
optimumscore=torch.tensor(fullpoints)
#normalise the optimum score
optimumscore=optimumscore/torch.norm(optimumscore,dim=-1,keepdim=True)
optimumscore=optimumscore@optimumscore.T

LossLabels=torch.arange(0,optimumscore.shape[0],device=optimumscore.device)
Loss=torch.nn.CrossEntropyLoss()
loss=Loss(optimumscore,LossLabels)
print("loss: ",loss)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
for i, key in enumerate(tokens.keys()):
    points=pca.transform(tokens[key])
    ax.scatter(points[:,0],points[:,1], label=key, alpha=0.5)

ax.set_title('2D PCA of Original and Adversarial Samples')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.legend()
plt.show() 

#save the pca plt
fig.savefig("PCA.png")