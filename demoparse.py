from test_tube import HyperOptArgumentParser
import os
class baseparser(HyperOptArgumentParser):
    def __init__(self,*args,strategy="random_search",**kwargs):

        super().__init__( *args,strategy=strategy, add_help=False) # or random search

        #our base parser looks like :
        self.TEMPDIR=os.path.join(os.getenv("TEMP","."),"results")
   
        self.opt_list("--save_freq", default=50, type=int, tunable=False)
        self.opt_list("--test_freq", default=3, type=int, tunable=False)
        self.opt_list("--weight_decay", default=0, type=float, tunable=False)
        self.opt_list("--warmup", default=0,options=[0,100,1000], type=int, tunable=False)
        self.opt_list("--momentum", default=0.9, type=float, tunable=False)
        self.opt_list("--test_batch_size", default=8, type=int, tunable=False)
        self.opt_list("--num_workers", default=32, type=int, tunable=False)
        self.opt_list("--epochs", default=8, type=int, tunable=False)
        self.opt_list("--learning_rate", default=5e-4, options=[5e-5,5e-4,1e-4], type=float, tunable=True) #originally 5e-5
        self.opt_list("--batch_size", default=96, options=[96], type=int, tunable=True)
        self.opt_list("--precision", default=32, type=int, options=[32,16], tunable=True)

        self.opt_list("--train_eps", default=1/255, options=[1/255,2/255,4/255,8/255], type=float, tunable=True)
        self.opt_list("--train_numsteps", default=5, options=[1,2,5,10], type=int, tunable=True)
        self.opt_list("--train_stepsize", default=1/255, type=float,options=[1/255,2/255,4/255,8/255], tunable=True)


        ####THESE ARE ACTUALLY JUST IGNORED
        self.opt_list("--test_eps", default=1/255, options=[1/255,2/255,4/255,8/255], type=float, tunable=False)
        self.opt_list("--test_numsteps", default=10, type=int, tunable=False)
        self.opt_list("--test_stepsize", default=1/255, options=[1/255,2/255,4/255,8/255], type=float , tunable=False)
        self.opt_list("--earlystop", default=1000, type=int, tunable=False)
        # model
        self.opt_list("--freeze_text",default=True,options=[True,False],type=bool,tunable=True)

        self.opt_list("--optimizer", default='sgd', type=str, options=["sgd","adam","adamw"],tunable=True)
        self.opt_list("--labelType", default='text', type=str, options= ["image","text","Modimage"],tunable=True)
        # dataset

        self.opt_list("--root", default=os.getenv("PWD","./data"), options=[os.getenv("PWD","/data")], type=str, tunable=False)
        self.opt_list("--dataset", default='cifar10', options=["coco","tinyImageNet"],type=str, tunable=True)
        self.opt_list("--image_size", default=224, type=int, tunable=False)
        # other
        self.opt_list("--attack_type", default="pgd", type=str,options=["pgd","cw","text","autoattack","noAttack"], tunable=False) # set this as tunable to trial different attack types
        self.opt_list("--test_attack_type", default="pgd", type=str,options=["pgd","cw","text","noAttack"], tunable=False)
        self.opt_list("--seed", default=0, type=int, tunable=False)
        self.opt_list("--model_dir", default=os.path.join(self.TEMPDIR,"modelckpts"), type=str, tunable=False)
        self.opt_list("--output_dir", default=self.TEMPDIR, type=str, tunable=False)
        self.opt_list("--filename", default=None, type=str, tunable=False)
        self.opt_list("--trial", default=1, type=int, tunable=False)
        self.opt_list("--resume", default=None, type=str, tunable=False)
        self.opt_list("--evaluate", default=False, action="store_true", tunable=False)
        self.opt_list("--debug", action='store_true', tunable=False)
        self.opt_list("--model", default='clip', type=str, tunable=False)
        self.opt_list("--imagenet_root", default=os.getenv("global_storage","./data"),options=[os.getenv("global_storage","./data")], type=str, tunable=False)
        self.opt_list("--arch", default='vit_b32', type=str, tunable=False)
        self.opt_list("--method", default='null_patch', type=str, options=['null_patch'], tunable=False)
        self.opt_list("--prompt_size", default=30, type=int, tunable=False)
        self.opt_list("--add_prompt_size", default=0, type=int, tunable=False)
        # self.opt_list("--CW", action='store_true', tunable=False)
        self.opt_list("--train_class_count", default=90, type=int, tunable=False)
        self.opt_list("--last_num_ft", default=-1, type=int, tunable=False)
        self.opt_list("--noimginprop", action='store_true', tunable=False)
        # self.opt_list("--autoattack", action='store_true', tunable=False)
        self.opt_list("--num_trials", default=0, type=int, tunable=False)
        #debug mode - We want to just run in debug mode...
        self.opt_list("--name", default="TestRun",options=["hecDeployment"], type=str, tunable=False)
        # self.argNames=["name","method","prompt_size","dataset","model","arch","learning_rate","weight_decay","batch_size","warmup","trial","add_prompt_size","optimizer", "freeze_text"]
        #self.opt_range('--neurons', default=50, type=int, tunable=True, low=100, high=800, nb_samples=8, log_base=None)
        
        #This is important when passing arguments as **config in launcher
        self.keys_of_interest=set("learning_rate dataset precision labelType batch_size train_eps train_numsteps train_stepsize attack_type prompt_size add_prompt_size optimizer freeze_text".split())

import wandb
from tqdm import tqdm



class parser(baseparser):
    def __init__(self,*args,strategy="random_search",**kwargs):

        super().__init__( *args,strategy=strategy, add_help=False,**kwargs) # or random search
        self.run_configs=set()
        self.keys=set()
    def generate_wandb_trials(self,entity,project):
        api = wandb.Api()

        runs = api.runs(entity + "/" + project)
        print("checking prior runs")
        for run in tqdm(runs):
            config=run.config
            sortedkeys=list([str(i) for i in config.keys() if i in self.keys_of_interest])
            sortedkeys.sort()
            values=list([str(config[i]) for i in sortedkeys])
            code="_".join(values)
            self.run_configs.add(code)
        hyperparams = self.parse_args()
        NumTrials=hyperparams.num_trials if hyperparams.num_trials>0 else 1
        trials=hyperparams.generate_trials(NumTrials)
        print("checking if already done...")
        trial_list=[]
        for trial in tqdm(trials):

            sortedkeys=list([str(i) for i in trial.__dict__.keys() if i in self.keys_of_interest])
            sortedkeys.sort()
            values=list([str(trial.__dict__[k]) for k in sortedkeys])
            
            code="_".join(values)
            while code in self.run_configs:
                trial=hyperparams.generate_trials(1)[0]
                sortedkeys=list([str(i) for i in trial.__dict__.keys() if i in self.keys_of_interest])
                sortedkeys.sort()
                values=list([str(trial.__dict__[k]) for k in sortedkeys])
            
                code="_".join(values)
            trial_list.append(trial)
        return trial_list
        
# Testing to check param outputs
if __name__== "__main__":
    
    #If you call this file directly, you'll see the default ARGS AND the trials that might be generated. 
    myparser=parser()
    hyperparams = myparser.parse_args()
    print(hyperparams.__dict__)

    for trial in hyperparams.generate_trials(hyperparams.num_trials):
        print("next trial")
        print(trial)
        
