import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import os
import time
#download the coco dataset
from torchvision.datasets import CocoCaptions
from torchvision.transforms import transforms
from PIL import Image
import zipfile
from torch.utils.data import Dataset, DataLoader
from pySmartDL import SmartDL as SmartDL



class HDF5COCOCaptionsDataset(CocoCaptions):
    def __init__(self, hdf5_file, **args):
        super().__init__(**args)
        self.file_path = hdf5_file
        self.file = h5py.File(hdf5_file, 'r')
        self.data = self.file['data']
        self.labels = self.file['labels']

    def __len__(self):
        return len(self.data)
    def _load_image(self, id: int) -> Image.Image:
        label = self.coco.loadImgs(id)[0]["file_name"]
        #load the data from the hdf5 file
        id=self.labels.index(label) 
        img = self.data[id]
        return img
        

class HDF5DataModule(pl.LightningDataModule):
    def __init__(self, file_path, batch_size=32, transform=None):
        super().__init__()
        self.HDF5file_path = os.path.join(file_path,"Dataset")
        #make HDF5 file at the given path
        #create the HDF5 file
        self.HDF5file = h5py.File(file_path, 'w')
        self.cache_dir=file_path
        self.HDF5file = h5py.File(file_path, 'r')
        self.batch_size = batch_size
        self.transform = transform
    def setup(self):
        self.dataset = HDF5COCOCaptionsDataset(self.HDF5file_path,root=self.root,annFile=self.annFile, transform=self.transform)

    def prepare_data(self):
        #Download dataset to the HDF5 file, using data and labels as keys
        self.train_dataset_dict={}
        self.train_text_names_dict={}

        if not os.path.exists(os.path.join(self.cache_dir,"annotations")):
            URLS=["http://images.cocodataset.org/zips/train2017.zip","http://images.cocodataset.org/annotations/annotations_trainval2017.zip"]
            
            for url in URLS:
                print("Downloading",url)
                obj=SmartDL(url,os.path.join(self.cache_dir,str(url).split('/')[-1]),progress_bar=False)
                obj.FileName=str(url).split('/')[-1]
                if not os.path.exists(obj.get_dest()):
                    obj.start(blocking=False)
                    print("obj Path ",obj.get_dest())
                while not obj.isFinished():
                    #print("Speed: %s" % obj.get_speed(human=True))
                    # print("Eta: %s" % obj.get_eta(human=True))
                    time.sleep(5)
                if obj.isSuccessful():
                    print("Downloaded: %s" % obj.get_dest())
                path = obj.get_dest()
                if obj.FileName.startswith("annotations"):
                    print("Extracting annotations")
                    print("path:",path)

                    with zipfile.ZipFile(path, 'r') as zip_ref:
                        try:
                            zip_ref.extractall(self.cache_dir)
                        except:
                            print("Error extracting annotations")
                            print("path:",path)
                            print("ann_dir:",self.ann_dir)
                else:
                    print("Extracting images")
                    print("path:",path)
                    if obj.FileName.endswith(".zip"):
                        print("Extracting zip")
                        with zipfile.ZipFile(path, 'r') as zip_ref:
                            try:
                                zip_ref.extractall(self.cache_dir)
                            except:
                                print("Error extracting images")
                                print("path:",path)
                                # print("data_dir:",self.data_dir)
                    print("Extracted: %s" % path)
        #now load the dataset
        self.annFile=os.path.join(self.cache_dir,"annotations","captions_train2017.json")
        self.instanceFile=os.path.join(self.cache_dir,"annotations","instances_train2017.json")
        self.root=os.path.join(self.cache_dir,"train2017")

        #copy all the images to the hdf5 file
        data=[]
        labels=[]
        print("Loading images")
        for i,filename in enumerate(filter(lambda x: x.endswith(".jpg"),os.listdir(self.root))):
        
            img=Image.open(os.path.join(self.root,filename))
            img=img.resize((224,224))
            data.append(torch.tensor(img))
            labels.append(filename)
            if i%1000==0:
                print("Loaded",i,"images")
                #save the data to the hdf5 file
                #if the dataset data already exists, extend it
                if 'data' in self.HDF5file.keys():
                    self.HDF5file['data'].resize((len(self.HDF5file['data'])+len(data),224,224,3))
                    self.HDF5file['data'][-len(data):]=data
                    self.HDF5file['labels'].resize((len(self.HDF5file['labels'])+len(labels)))
                    self.HDF5file['labels'][-len(labels):]=labels
                else:
                    self.HDF5file.create_dataset('data', data=data)
                    self.HDF5file.create_dataset('labels', data=labels)
                print("Loaded",len(data),"images")
                data=[]
                labels=[]
        #remove the images from the self.root
        print("Removing images")
        for filename in filter(lambda x: x.endswith(".jpg"),os.listdir(self.root)):
            os.remove(os.path.join(self.root,filename))


    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)
    

if __name__ == '__main__':
    #download the coco dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=os.getcwd())
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    data_dir = args.data_dir
    batch_size = args.batch_size
    dm = HDF5DataModule(data_dir, batch_size)
    dm.prepare_data()
    dm.setup()
    print("Data prepared")
    print("Data loaded")
    dataloader=dm.train_dataloader()
    for i, (img, label) in enumerate(dataloader):
        print(i, img.shape, label)
        if i==2:
            break