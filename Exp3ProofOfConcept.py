#This file is going to be a proof of concept graph plotting colour coded points. 


# import COCO datamodule

import matplotlib.pyplot as plt
import numpy as np
import torch
from COCODataModule import MyDataModule
import clip
DEVICE="cuda"
Vocab_size = 49403
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
@torch.no_grad()
def BERTSCORE(captions,Noise_captions):
    global Vocab_size
    #captions is shape B,77
    #Noise_captions is shape B,B,77
    #this function is going to return a B,B tensor of bert scores
    captions=captions.squeeze(1) #shape B,77
    captions,_EOTS=encode_text(captions) #shape B,77,512
    Vocab_size=max(Vocab_size,captions.max().item())
    Noise_captions,CaptEOTS=encode_text(Noise_captions.flatten(0,1)) #shape B,77,77,512
    CaptEOTS=CaptEOTS.view(B,77,512)
    Noise_captions=Noise_captions.view(B,77,77,512) #shape 77,B*B,512
    captions=captions.unsqueeze(1) #shape B,1,77,512
    captions=captions/ torch.norm(captions, dim=-1, keepdim=True)
    Noise_captions=Noise_captions/ torch.norm(Noise_captions, dim=-1, keepdim=True)

    sim = (captions @ Noise_captions.transpose(-1, -2)).squeeze(-1)
    #sim shape is B,77,77,77

    word_precision = sim.max(dim=-2)[0]
    word_recall = sim.max(dim=-1)[0]
    
    P = (word_precision).sum(dim=-1) #shape B,B
    R = (word_recall).sum(dim=1) #shape B,B
    F = 2 * word_precision * word_recall / (word_precision + word_recall)
    F=F.sum(dim=-1)
    # print(F.shape)#assumed shape B,B
    return (P,R,F),_EOTS,CaptEOTS


if __name__ == "__main__":
    from tqdm import tqdm
    import argparse
    import os 
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--imagenet_root', type=str, default='./data')
    parser.add_argument('--tinyimagenet_root', type=str, default='./data')
    parser.add_argument('--cache_dir', type=str, default='./data')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    filelist=os.listdir()
    BertFiles=[file for file in filelist if "BertScore_" in file]
    if len(BertFiles)>0:
        print("BertScore files already exist, skipping")
    else:
        dm = MyDataModule(Cache_dir=args.cache_dir, dataset=args.dataset, batch_size=args.batch_size, imagenet_root=args.imagenet_root, tinyimagenet_root=args.tinyimagenet_root, debug=args.debug)
        dm.train_dataset_names = ["coco"]
        dm.prepare_data()
        dm.setup()
        print("Calculating BertScores")
        for idx,input in tqdm(enumerate(dm.train_dataloader())):
            _, _, captions = input
            #Ignore the image component, we just care about the target
            #make a B sized array of random numbers
            captions=captions.to(DEVICE)
            B =captions.shape[0]

            noise= torch.randint(-Vocab_size, Vocab_size, (1,77,1),device=DEVICE,dtype=torch.long)
            
            #add the 2 to get the shape B,B,77
            Noise_captions = torch.add(captions, noise) #B,77,77
            Noise_captions = torch.remainder(Noise_captions,Vocab_size) #B,77,77
            #
            scores,caption_Encs,Noise_captions_Encs=BERTSCORE(captions,Noise_captions)#[B,B] and range (-1,1)
            scores=scores[2]
        
            #calculate Betrscoresbetween the GT and the noise caption
            ground_truth.append(caption_Encs.cpu())
            AttackedCaptions.append(Noise_captions_Encs.cpu())
            BertScores.append(scores.cpu())
            if idx % 1000 == 0:
                ground_truth=torch.cat(ground_truth,0).detach().cpu().numpy()
                AttackedCaptions=torch.cat(AttackedCaptions,0).flatten(0,1).detach().cpu().numpy()
                BertScores=torch.cat(BertScores,0).flatten(0,1).detach().cpu().numpy()
                #save lists to a npz file 
                np.savez("BertScore_{}.npz".format(idx),ground_truth=ground_truth,AttackedCaptions=AttackedCaptions,BertScores=BertScores)
                ground_truth=[]
                AttackedCaptions=[]
                BertScores=[]
                

    #plot the data
    #train PCA on the GT data

    PCA_model = make_pipeline(StandardScaler(), PCA(n_components=2))
    Filelist=os.listdir()
    BertFiles=[file for file in Filelist if "BertScore_" in file]
    for file in BertFiles:
        contents=np.load(file)
        ground_truth=contents["ground_truth"]
        PCA_model.fit(ground_truth)


    for file in BertFiles:
        contents=np.load(file)
        ground_truth=contents["ground_truth"]
        AttackedCaptions=contents["AttackedCaptions"]
        #
        # repeat each element in ground_truth (1,2,3,4) to (1,1,1,2,2,2,3,3,3,4,4,4) to match the shape of the BertScores
        repeated_ground_truth = np.repeat(ground_truth, 77, axis=0)
        idx=np.arange(0,ground_truth.shape[0]*77)
        idx=np.mod(idx,77)
        distances = np.linalg.norm(repeated_ground_truth - AttackedCaptions, axis=1)
        normed_repeated_ground_truth = repeated_ground_truth / np.linalg.norm(repeated_ground_truth, axis=1)[:, None]
        normed_AttackedCaptions = AttackedCaptions / np.linalg.norm(AttackedCaptions, axis=1)[:, None]
        cosine_similarities = np.sum(normed_repeated_ground_truth * normed_AttackedCaptions, axis=1)

        BertScores=contents["BertScores"]
        #plot bertscore by index
        fig, ax = plt.subplots()
        ax.scatter(idx, BertScores, c="Blue",label="BertScores")
        ax.scatter(idx, cosine_similarities, c="Red",label="Cosine Similarity")
        ax.scatter(idx, distances, c="Green",label="Cartesian Distance")
        plt.title("Distance to GT by noise index")
        plt.xlabel("Index")
        plt.ylabel("BertScores")
        plt.savefig("BertScoresbyIndex.png")
        plt.legend()



        #plot the bertscores against cosine similarity
        fig, ax = plt.subplots()
        ax.scatter(distances, BertScores, c=cosine_similarities)
        plt.title("BertScores vs cartesian distance")
        plt.xlabel("Cartesian Distance")
        plt.ylabel("BertScores")
        plt.savefig("BertScoresvCartesian.png")

        


        fig, ax = plt.subplots()
        ax.scatter(cosine_similarities, BertScores, c=distances)
        plt.title("BertScores vs Cosine Similarity")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("BertScores")
        plt.savefig("BertScoresvCosine.png")

        fig, ax = plt.subplots()
        ax.scatter(distances, cosine_similarities, c=BertScores)
        plt.title("Cosine Similarity vs Cartesian Distance")
        plt.xlabel("Cartesian Distance")
        plt.ylabel("Cosine Similarity")
        plt.savefig("CosinevCartesian.png")





        for i in range(0,len(ground_truth),10):
            fig, ax = plt.subplots()

            GT=PCA_model.transform(ground_truth[i:i+10])
            AttackedCaptions=PCA_model.transform(AttackedCaptions[i*77:(i+10)*77])
            scores=BertScores[i*77:(i+10)*77]
            ax.scatter(GT[:,0],GT[:,1],c="Green",label="Ground Truth")
            ax.scatter(AttackedCaptions[:,0],AttackedCaptions[:,1],c=scores,label="Attacked Captions")
            #add a legend
            plt.legend()
            plt.title("PCA of Ground Truth and Attacked Captions")
            plt.savefig("PCALabelsAttackes{}.png".format(i))
            # plt.show()

        #plot the data

   



#Next steps:
'''
Are there patterns the model gets wrong?
Word types? #(assessed by token idx in problem cases)
are there glitch tokens? 
IS PCA not clear enough? Should we look for data heuristics instead? 

Are there ideas that the model consistently fumbles (SOT /EOT errors)



'''