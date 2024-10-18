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
Loss=torch.nn.CrossEntropyLoss()

tokens={}
model, preprocess = clip.load("ViT-B/32",device='cpu')
with torch.inference_mode(True):
    for key in class_names.keys():
        names=class_names[key]
        print("datasets: ",key)
        print("names: ",names)
        names=clip.tokenize(names)
        tokens.update({key:model.encode_text(names)})

fullpoints=torch.cat(tuple(list(tokens.values())),axis=0)
optimumscore=fullpoints/torch.norm(fullpoints,dim=-1,keepdim=True)
optimumscore=optimumscore@optimumscore.T
##plot this as a confusion matrix
LossLabels=torch.arange(0,optimumscore.shape[0],device=optimumscore.device)
loss=Loss(optimumscore,LossLabels)

print("loss: ",loss)
plt.matshow(optimumscore)
plt.title('Confusion Matrix of Original Classes, optimal score is '+str(loss.item()))
plt.savefig("confusion_matrix.png")

LossByBatchSize={}
#I want to show the minimum score by batch size by taking a random sample of the vectors...
for batchsize in [2,4,8,16,32,64,128,256,512]:
    LossLabels=torch.arange(0,batchsize,device=optimumscore.device)
    scores=[]
    for i in range(200):
        randomindices=torch.randperm(optimumscore.shape[0])[:batchsize]
        selection=fullpoints[randomindices]
        selection=selection/torch.norm(selection,dim=-1,keepdim=True)
        selection=selection@selection.T
        scores.append(Loss(selection,LossLabels).item())
    LossByBatchSize.update({batchsize:np.mean(scores)})


#plot the loss by batch size
#new plot
plt.figure()
plt.plot(list(LossByBatchSize.keys()),list(LossByBatchSize.values()))
plt.title('Minimum Expected Loss by Batch Size')
#use log scale on X axis
plt.xscale('log')
plt.xlabel('Batch Size')
plt.ylabel('Loss')
plt.show()
plt.savefig("batchsize_loss.png")


X_pca = pca.fit_transform(fullpoints.detach().cpu().numpy())
optimumscore=fullpoints
#normalise the optimum score

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