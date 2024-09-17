

from pytorch_lightning import LightningModule
import torch.nn as nn
import torch

from utils import cosine_lr
from utils import one_hot_embedding
from utils import accuracy,clamp,normalize
import torch.nn.functional as F
from clip import clip
from models.prompters import TokenPrompter, NullPrompter
from autoattack import AutoAttack
from utils import clip_img_preprocessing


def multiGPU_CLIP(model, images, text_tokens, prompt_token):
    prompt_token = prompt_token.repeat(images.size(0), 1, 1)

    img_embed, scale_text_embed = model(images, text_tokens, prompt_token)
    img_embed_norm = img_embed / img_embed.norm(dim=-1, keepdim=True)
    scale_text_embed_norm = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
    logits_per_image = img_embed_norm @ scale_text_embed_norm.t()
    #logits_per_text = scale_text_embed_norm @ img_embed_norm.t()
    return logits_per_image#, logits_per_text, img_embed, scale_text_embed


def multiGPU_CLIP_NP(model, images, text_tokens):
    img_embed, scale_text_embed = model(images, text_tokens, None)
    img_embed_norm = img_embed / img_embed.norm(dim=-1, keepdim=True)
    scale_text_embed_norm = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
    logits_per_image = img_embed_norm @ scale_text_embed_norm.t()
    #logits_per_text = scale_text_embed_norm @ img_embed_norm.t()
    return logits_per_image#, logits_per_text, img_embed, scale_text_embed



class myLightningModule(LightningModule):
    '''
    This training code follows the standard structure of Pytorch - lighthning. It's worth looking at their docs for a more in depth dive as to why it is this was
    '''
    
    def __init__(self,
                args,
                ):

        super().__init__()
        self.save_hyperparameters()
        self.loss=torch.nn.CrossEntropyLoss()
        self.args = args
        add_prompt_len = 0 if args.add_prompt == 'none' else 1
        self.upper_limit, self.lower_limit = 1, 0
        self.model, _ = clip.load('ViT-B/32', device=self.device, jit=False, prompt_len=add_prompt_len)
        self.model_ori, _ = clip.load('ViT-B/32', device=self.device, jit=False, prompt_len=add_prompt_len)
        self.model_text, _= None, None
        self.prompter = NullPrompter()
        self.add_prompter = TokenPrompter(add_prompt_len)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction="sum")

        if args.norm=='l_inf':
            self.init_delta=self.init_uniform
            self.clamp=self.clamp_inf
        elif args.norm=='l_2':
            self.init_delta=self.init_normal
            self.clamp=self.clamp_2
        else:
            raise ValueError
        if not args.Noattack:
            self.attack=self.no_attack

    def init_uniform(self, X,eps):
        delta=  torch.zeros_like(X,device=self.device,).uniform_(-eps, eps)
        delta = clamp(delta, self.lower_limit - X, self.upper_limit - X)
        delta.requires_grad = True
        return delta
    
    def init_normal(self, X,eps):
            delta=torch.zeros_like(X,device=self.device)
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * eps
            delta = clamp(delta, self.lower_limit - X, self.upper_limit - X)
            delta.requires_grad = True
            return delta
    
    def clamp_inf(self,d,alpha,g,eps):
        return torch.clamp(d + alpha * torch.sign(g), min=-eps, max=eps)
    
    def clamp_2(self,d,alpha,g,eps):
        g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
        scaled_g = g / (g_norm + 1e-10)
        d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=eps).view_as(d)
        return d
    
    

    def attack_pgd(self,  X, target, text_tokens, alpha, attack_iters, restarts=1, early_stop=True, epsilon=0):
        delta=self.init_delta(X,epsilon)
        for _ in range(attack_iters):
            # output = model(normalize(X ))
            prompted_images = self.prompter(normalize(X + delta))
            prompt_token = self.add_prompter()
            output = multiGPU_CLIP(self.model, prompted_images, text_tokens, prompt_token)
            loss = self.criterion(output, target)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            x = X[:, :, :, :]
            d=self.clamp(d,alpha,g,epsilon)
            d = clamp(d, self.lower_limit - x, self.upper_limit - x)
            delta.data[:, :, :, :] = d
            delta.grad.zero_()
        return delta
        
    def attack_pgd_noprompt(self, X, target, text_tokens, alpha, attack_iters, restarts=1, early_stop=True, epsilon=0):
        delta=self.init_delta(X,epsilon)
        for _ in range(attack_iters):
            _images = normalize(X + delta)
            output= multiGPU_CLIP_NP( self.model, _images, text_tokens)
            loss = self.criterion(output, target)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            x = X[:, :, :, :]
            d=self.clamp(d,alpha,g,epsilon)
            d = clamp(d, self.lower_limit - x, self.upper_limit - x)
            delta.data[:, :, :, :] = d
            delta.grad.zero_()

        return delta


    def attack_CW(self, X, target, text_tokens, alpha,attack_iters, restarts=1, early_stop=True, epsilon=0):
        delta=self.init_delta(X,epsilon)

        for _ in range(attack_iters):
            # output = model(normalize(X ))
            prompted_images = self.prompter(normalize(X + delta))
            prompt_token = self.add_prompter()
            output= multiGPU_CLIP(self.model, prompted_images, text_tokens, prompt_token)
            label_mask = one_hot_embedding(target, output.size(1))
            correct_logit = torch.sum(label_mask * output, dim=1)
            wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, axis=1)
            # loss = criterion(output, target)
            loss = - torch.sum(F.relu(correct_logit - wrong_logit + 50))
            loss.backward()
            grad = delta.grad.detach()
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            x = X[:, :, :, :]
            d=self.clamp(d,alpha,g,epsilon)
            d = clamp(d, self.lower_limit - x, self.upper_limit - x)
            delta.data[:, :, :, :] = d
            delta.grad.zero_()
        return delta

    def attack_CW_noprompt(self, X, target, text_tokens, alpha, attack_iters, restarts=1, early_stop=True, epsilon=0):
        delta=self.init_delta(X,epsilon)
        for _ in range(attack_iters):
            # output = model(normalize(X ))
            _images = normalize(X + delta)
            # output, _ = model(_images, text_tokens)
            output= multiGPU_CLIP_NP(self.model, _images, text_tokens)
            label_mask = one_hot_embedding(target, output.size(1))
            correct_logit = torch.sum(label_mask * output, dim=1)
            wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, axis=1)
            # loss = criterion(output, target)
            loss = - torch.sum(F.relu(correct_logit - wrong_logit + 50))
            loss.backward()
            grad = delta.grad.detach()
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            x = X[:, :, :, :]
            d=self.clamp(d,alpha,g,epsilon)
            d = clamp(d, self.lower_limit - x, self.upper_limit - x)
            delta.data[:, :, :, :] = d
            delta.grad.zero_()
        return delta

    def attack(self, images, target, text_tokens, alpha, attack_iters, epsilon=0):
            delta = self.attack_pgd( images, target, text_tokens, alpha, attack_iters, epsilon=self.args.train_eps)
            return images+delta
    
    def no_attack(self, images, *args, **kwargs):
            return images

    def forward(self,input):
        #This inference steps of a foward pass of the model 
        return self.model(input)
    
    def training_step(self, batch, batch_idx):
        #The batch is collated for you, so just seperate it here and calculate loss. 
        #By default, PTL handles optimization and scheduling and logging steps. so All you have to focus on is functionality. Here's an example...
        images, text, target = batch
        prompted_clean_images = self.prompter(images)
        images=self.attack(images, target, text, self.args.alpha, self.args.attack_iters, epsilon=self.args.train_eps)
        prompted_images = self.prompter(normalize(images))
        output= multiGPU_CLIP_NP( self.model, prompted_images, text)
        output_ori= multiGPU_CLIP_NP( self.model_ori, prompted_images, text)
        output_clean = multiGPU_CLIP_NP( self.model, prompted_clean_images, text)

        loss_advori = self.criterion_kl(F.log_softmax(output, dim=1), F.softmax(output_ori, dim=1))
        loss_advclean = self.criterion_kl(F.log_softmax(output, dim=1), F.softmax(output_clean, dim=1))
        loss = self.criterion(output, target) + loss_advclean + loss_advori
        
        self.model.module.logit_scale.data = torch.clamp(self.model.module.logit_scale.data, 0, 4.6052)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
   

    def on_train_epoch_end(self):
        l2_norm_obj = sum(p.norm(2) for p in self.model.module.visual.parameters())
        l2_norm_ori = sum(p.norm(2) for p in self.model_ori.module.visual.parameters())
        ratio = abs(l2_norm_ori - l2_norm_obj) / float(l2_norm_ori)
        abs_l2 = abs(l2_norm_ori - l2_norm_obj)
        self.log('l2_norm_obj', l2_norm_obj, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('l2_norm_ori', l2_norm_ori, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('ratio', ratio, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('abs_l2', abs_l2, on_step=False, on_epoch=True, prog_bar=True, logger=True)
      
    def validation_step(self, batch, batch_idx):
      
        images,text,target=batch
        prompt_token = None
        output_prompt= multiGPU_CLIP_NP(self.model, self.prompter(images), text)

        loss = self.criterion(output_prompt, target)

        # measure accuracy and record loss
        acc1 = accuracy(output_prompt, target, topk=(1,))
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc1[0].item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.args.CW:
            delta_prompt = self.attack_CW(
                                    images, target, text,
                                    self.args.test_stepsize, self.args.test_numsteps, epsilon=self.args.test_eps)
        elif self.args.autoattack:
            def model_fn(x):
                output_a = multiGPU_CLIP_NP(self.model, self.prompter(clip_img_preprocessing(x)),text)
                return output_a.to(torch.float32)

            adversary = AutoAttack(model_fn, norm='Linf', eps=self.args.test_eps, version='standard')
            adv_samples = adversary.run_standard_evaluation(images, target, bs=100)
            delta_prompt = adv_samples - images
            delta_prompt = clamp(delta_prompt, self.lower_limit - images, self.upper_limit - images)
        else:
            delta_prompt = self.attack_pgd(images, target, text,self.args.test_stepsize, self.args.test_numsteps, epsilon=self.args.test_eps)

        prompt_token = self.add_prompter()
        # output_prompt_adv, _ = model(prompter(clip_img_preprocessing(images + delta_prompt)), text_tokens, prompt_token)
        output_prompt_adv = multiGPU_CLIP( self.model,
                                                    self.prompter(clip_img_preprocessing(images + delta_prompt)),
                                                    text, prompt_token)

        loss = self.criterion(output_prompt_adv, target)

        # bl attack
        torch.cuda.empty_cache()

        # measure accuracy and record loss
        acc1 = accuracy(output_prompt_adv, target, topk=(1,))
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc1[0].item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        acc1 = accuracy(output_prompt_adv, target, topk=(1,))
        self.log('top1_adv_org', acc1[0].item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

         #You could log here the val_loss, or just print something. 
        
    def configure_optimizers(self):
        # pretty sure we probably want to use the same optimizer as the original paper: the adamw optimizer
        # https://pytorch.org/docs/stable/optim.html#torch.optim.AdamW
        # https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html

        optimizer = torch.optim.SGD(list(self.model.module.visual.parameters())[-self.args.last_num_ft:],
                                        lr=self.args.learning_rate,
                                        momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay)
        

        if self.args.last_num_ft == -1:
            optimizer = torch.optim.SGD(self.model.module.visual.parameters(),
                                        lr=self.args.learning_rate,
                                        momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay)
        scheduler = cosine_lr(optimizer, self.args.learning_rate, self.args.warmup, self.args.total_steps)
        return [optimizer,scheduler]
