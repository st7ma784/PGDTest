#This file is going to be a proof of concept graph plotting colour coded points. 


# import COCO datamodule

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import torch
from COCODataModule import MyDataModule
import clip
DEVICE="cuda"
Vocab_size = 49207
clip_model,preprocess=clip.load("ViT-B/32", device=DEVICE, jit=False)
ground_truth=[]
AttackedCaptions=[]
BertScores=[]
def encode_text(text):
    x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)
    EOTs= x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

    return x,EOTs # [batch_size, n_ctx, d_model]
def BERTSCORE(captions,Noise_captions):
    Encoder=clip_model.encoder
    #captions is shape B,77
    #Noise_captions is shape B,B,77
    #this function is going to return a B,B tensor of bert scores
    captions,_EOTS=encode_text(captions) #shape B,77,512
    Noise_captions,CaptEOTS=encode_text(Noise_captions.flatten(0,1)) #shape B,B,77,512
    Noise_captions=Noise_captions.view(B,B,77,512).permute(0,2,1) #shape 77,B*B,512
    captions=captions.unsqueeze(1) #shape B,1,77,512
    captions=captions/ torch.norm(captions, dim=-1, keepdim=True)
    Noise_captions=Noise_captions/ torch.norm(Noise_captions, dim=-1, keepdim=True)

    sim = (captions @ Noise_captions.transpose(-1, -2)).squeeze(-1)    
    #sim shape is B,B,77,77
    word_precision = sim.max(dim=-2)[0]
    word_recall = sim.max(dim=-1)[0]
    
    P = (word_precision).sum(dim=-1) #shape B,B
    R = (word_recall).sum(dim=1) #shape B,B
    F = 2 * word_precision * word_recall / (word_precision + word_recall)
    F=F.sum(dim=-1)
    print(F.shape)#assumed shape B,B
    return (P,R,F),_EOTS,CaptEOTS.unflatten(0,(B,B,512))


if __name__ == "__main__":
    from tqdm import tqdm
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--imagenet_root', type=str, default='./data')
    parser.add_argument('--tinyimagenet_root', type=str, default='./data')
    parser.add_argument('--cache_dir', type=str, default='./data')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    dm = MyDataModule(Cache_dir=args.cache_dir, dataset=args.dataset, batch_size=args.batch_size, imagenet_root=args.imagenet_root, tinyimagenet_root=args.tinyimagenet_root, debug=args.debug)
    for input in tqdm(dm.train_dataloader()):
        _, _, captions = input
        #Ignore the image component, we just care about the target
        #make a B sized array of random numbers
        captions=captions.to(DEVICE)
        B =captions.shape[0]

        noise= torch.randint(-Vocab_size, Vocab_size, (1,B,),device=DEVICE,dtype=torch.float32)
        #print(noise.shape) #torch.Size([B])
        noise=noise.unsqueeze(0)
        captions=captions.unsqueeze(1)
        
        #add the 2 to get the shape B,B,77
        Noise_captions = torch.add(captions, noise) 
        Noise_captions = torch.clamp(Noise_captions, 0, Vocab_size)
        scores,caption_encodes=BERTSCORE(captions,Noise_captions)#[B,B] and range (-1,1)

        noise_caption_location=clip_model.encode_text(noise_caption_location.flatten(0,1))
        #print(REsults.shape)
        noise_caption_location=noise_caption_location.view(B,B,512)
        
        #calculate Betrscoresbetween the GT and the noise caption
        ground_truth.append(caption_encodes)
        AttackedCaptions.append(noise_caption_location)
        BertScores.append(scores)


    #plot the data
    #train PCA on the GT data
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    PCA_model = make_pipeline(StandardScaler(), PCA(n_components=2))
    GT=torch.cat(ground_truth,0).detach().cpu().numpy()
    AttackedCaptions=torch.cat(AttackedCaptions,0).flatten(0,1).detach().cpu().numpy()
    BertScores=torch.cat(BertScores,0).flatten(0,1).detach().cpu().numpy()

    PCA_model.fit(GT)
    GT=PCA_model.transform(GT)
    AttackedCaptions=PCA_model.transform(AttackedCaptions)

    #plot the data
    fig, ax = plt.subplots()
    ax.scatter(GT[:,0],GT[:,1],c="Green",label="Ground Truth")
    ax.scatter(AttackedCaptions[:,0],AttackedCaptions[:,1],c=BertScores,label="Attacked Captions")

    #add a legend
    plt.legend()
    plt.title("PCA of Ground Truth and Attacked Captions")
    plt.savefig("PCALabelsAttackes.png")
    plt.show()




#Next steps:
'''
Are there patterns the model gets wrong?
Word types? #(assessed by token idx in problem cases)
are there glitch tokens? 
IS PCA not clear enough? Should we look for data heuristics instead? 

Are there ideas that the model consistently fumbles (SOT /EOT errors)



'''