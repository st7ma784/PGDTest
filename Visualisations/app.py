import torch
from flask import Flask, render_template, request, jsonify, send_file, make_response
from nargsLossCalculation import get_loss_fn
from functools import reduce
from io import BytesIO
import zipfile
from PIL import Image

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

app = Flask(__name__,template_folder='.')



@app.route("/") 
def index():
    return render_template("./index.html") #the index page is a menu with 2 buttons that lead to the 2 pages

@app.route("/2d")
def Pagewith2Dpoints():
    return render_template("./2d.html")

@app.route("/ImagePage")
def ImagePage():
    return render_template("./ImagePage.html") 


def pgd_attack(model, images, labels, eps=0.4, alpha=0.2, iters=2, clip_min=-1, clip_max=1):
    """
    Performs a PGD attack on a batch of points.

    Args:
        model: A function that takes a batch of points and returns predictions (e.g., a linear model).
        images: The batch of points (numpy array).
        labels: The true labels for the points (list).
        eps: Maximum perturbation magnitude.
        alpha: Step size for the attack.
        iters: Number of attack iterations.
        clip_min: Minimum value for each coordinate.
        clip_max: Maximum value for each coordinate.

    Returns:
        A batch of adversarially perturbed points.
    """
    perturbed_images = np.copy(images)
    for _ in range(iters):
        perturbed_images_tensor = torch.tensor(perturbed_images, dtype=torch.float,requires_grad=True)
        model_output = model(perturbed_images_tensor)  # Placeholder - replace with your actual model

        # Placeholder: Replace with your loss function based on model output and labels
        loss = torch.nn.CrossEntropyLoss()(model_output, torch.tensor(labels))

        loss.backward()
        grad = perturbed_images_tensor.grad.detach().numpy()
        perturbation = alpha * np.sign(grad)
        perturbed_images = perturbed_images + perturbation
        perturbed_images = np.clip(perturbed_images, images - eps, images + eps)
        perturbed_images = np.clip(perturbed_images, clip_min, clip_max)

    return perturbed_images

@app.route('/2d/data', methods=['GET','POST'])
async def perturbPoints():
    data=request.get_json()
    wh=torch.tensor([[data['width'],data['height']]])
    x=[float(x[:-2]) for x in filter(lambda a: a != '',data['x'])]
    y=[float(y[:-2]) for y in filter(lambda a: a != '',data['y'])]
    labels=[int(l) for l in data['labels']]
    xys=[(torch.tensor([[x,y]],requires_grad=False)-(wh/2))/wh for x,y in zip(x,y)]
    
    normed=data['norm']
    alpha=data['alpha']
    epsilon=data['epsilon']
    iters=data['iters']
    if normed:
        xys=xys/torch.norm(xys,keepdim=True,dim=1)
    classes=torch.unique(labels)
    #make linear model with 2 inputs and number of classes
    model=torch.nn.Linear(2,classes.size(0))
    #fit model to points
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
    criterion=torch.nn.CrossEntropyLoss()
    xys=torch.stack([torch.tensor([[x,y]],requires_grad=False)for x,y in zip(x,y)])-(wh/2)
    xys=xys/wh
    for _ in range(100):
        optimizer.zero_grad()
        output=model(xys)
        loss=criterion(output,labels)
        loss.backward()
        optimizer.step()
    perturbed_points=pgd_attack(model, xys, labels, eps=epsilon, alpha=alpha, iters=iters, clip_min=-1, clip_max=1)
    return perturbed_points.tolist()

@app.route('/2d/classifyPoints', methods=['GET','POST'])
async def fitmodeltoPoints():
    data=request.get_json()
    wh=torch.tensor([[data['width'],data['height']]])
    x=[float(x[:-2]) for x in filter(lambda a: a != '',data['x'])]
    y=[float(y[:-2]) for y in filter(lambda a: a != '',data['y'])]
    labels=torch.tensor(data['labels'])
    classes=torch.unique(labels)
    #make linear model with 2 inputs and number of classes
    layer=torch.nn.Linear(2,classes.size(0))
    #fit model to points
    optimizer=torch.optim.Adam(layer.parameters(),lr=0.01)
    criterion=torch.nn.CrossEntropyLoss()
    xys=torch.stack([torch.tensor([[x,y]],requires_grad=False)for x,y in zip(x,y)])-(wh/2)
    xys=xys/wh
    for _ in range(100):
        optimizer.zero_grad()
        output=layer(xys)
        loss=criterion(output,labels)
        loss.backward()
        optimizer.step()
    return layer.weight.tolist(),layer.bias.tolist()



if __name__ == "__main__":

        
    app.run(host="0.0.0.0", port=8000 )



