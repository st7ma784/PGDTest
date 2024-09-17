

from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
from functools import partial
from typing import Optional
from warnings import warn
from utils import cosine_lr
from utils import one_hot_embedding
from utils import accuracy,clamp,clip_img_preprocessing


      

          


def attack_CW(prompter, model, model_text, model_image, add_prompter, criterion, X, target, text_tokens, alpha,
              attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        # output = model(normalize(X ))

        prompted_images = prompter(clip_img_preprocessing(X + delta))
        prompt_token = add_prompter()

        output= multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)

        num_class = output.size(1)
        label_mask = one_hot_embedding(target, num_class)
        label_mask = label_mask.cuda()

        correct_logit = torch.sum(label_mask * output, dim=1)
        wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, axis=1)

        # loss = criterion(output, target)
        loss = - torch.sum(F.relu(correct_logit - wrong_logit + 50))

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta


def attack_CW_noprompt(prompter, model, model_text, model_image, criterion, X, target, text_tokens, alpha,
                       attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        # output = model(normalize(X ))

        _images = clip_img_preprocessing(X + delta)
        # output, _ = model(_images, text_tokens)

        output= multiGPU_CLIP(model, _images, text_tokens, None)

        num_class = output.size(1)
        label_mask = one_hot_embedding(target, num_class)
        label_mask = label_mask.cuda()

        correct_logit = torch.sum(label_mask * output, dim=1)
        wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, axis=1)

        # loss = criterion(output, target)
        loss = - torch.sum(F.relu(correct_logit - wrong_logit + 50))

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta


def attack_pgd(prompter, model, model_text, model_image, add_prompter, criterion, X, target, text_tokens, alpha,
               attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        # output = model(normalize(X ))

        prompted_images = prompter(clip_img_preprocessing(X + delta))
        prompt_token = add_prompter()

        output = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)

        loss = criterion(output, target)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta




def multiGPU_CLIP(model, images, text_tokens, prompt_token=None):
    if prompt_token is not None:
        bs = images.size(0)
        prompt_token = prompt_token.repeat(bs, 1, 1)

    img_embed, scale_text_embed = model(images, text_tokens, prompt_token)
    img_embed_norm = img_embed / img_embed.norm(dim=-1, keepdim=True)
    scale_text_embed_norm = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
    logits_per_image = img_embed_norm @ scale_text_embed_norm.t()
    #logits_per_text = scale_text_embed_norm @ img_embed_norm.t()
    return logits_per_image#, logits_per_text, img_embed, scale_text_embed




from clip import clip

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
        #Define your own model here, 
        self.args = args

        self.model, preprocess = clip.load('ViT-B/32', device=self.device, jit=False, prompt_len=add_prompt_len)
        self.model_ori, preprocess_ori = clip.load('ViT-B/32', device=self.device, jit=False, prompt_len=add_prompt_len)
        self.model_text, model_image = None, None

        self.prompter = NullPrompter()
        self.add_prompter = TokenPrompter(add_prompt_len)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction="sum")

    def attack_pgd(self,  X, target, text_tokens, alpha, attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
        delta = torch.zeros_like(X,device=self.device)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon

        delta = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            # output = model(normalize(X ))
            prompted_images = self.prompter(clip_img_preprocessing(X + delta))
            prompt_token = self.add_prompter()
            output = multiGPU_CLIP(self.model, prompted_images, text_tokens, prompt_token)
            loss = self.criterion(output, target)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            x = X[:, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[:, :, :, :] = d
            delta.grad.zero_()
        return delta
        
    def attack_pgd_noprompt(self, X, target, text_tokens, alpha, attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
        delta = torch.zeros_like(X,device=self.device)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            _images = clip_img_preprocessing(X + delta)
            output= multiGPU_CLIP( self.model, _images, text_tokens, None)

            loss = self.criterion(output, target)

            loss.backward()
            grad = delta.grad.detach()
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            x = X[:, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[:, :, :, :] = d
            delta.grad.zero_()

        return delta


    def forward(self,input):
        #This inference steps of a foward pass of the model 
        return self.model(input)

    def on_train_epoch_start(self):
        #This is called at the start of each epoch. 
        pass


    def training_step(self, batch, batch_idx):
        #The batch is collated for you, so just seperate it here and calculate loss. 
        #By default, PTL handles optimization and scheduling and logging steps. so All you have to focus on is functionality. Here's an example...
        images, target = batch
        BATCH_SIZE = images.size(0)

        
        # text_tokens = clip.tokenize(texts_train) # now this is in our dataloader ?>!?

        tem_clean = clip_img_preprocessing(images)

        if not args.Noattack:
            delta = self.attack_pgd( images, target, text_tokens, alpha, attack_iters, 'l_inf', epsilon=self.args.train_eps)
            images=images+delta
        
        tmp = clip_img_preprocessing(images)

        prompted_images = prompter(tmp)
        prompted_clean_images = prompter(tem_clean)
        prompt_token = None

        output= multiGPU_CLIP( model, prompted_images, text_tokens,
                                                prompt_token)
        output_ori= multiGPU_CLIP( model_ori, prompted_images,
                                                        text_tokens,
                                                        prompt_token)
        output_clean = multiGPU_CLIP( model, prompted_clean_images,
                                                            text_tokens,
                                                            prompt_token)

        loss_advori = criterion_kl(F.log_softmax(output, dim=1), F.softmax(output_ori, dim=1))
        loss_advclean = criterion_kl(F.log_softmax(output, dim=1), F.softmax(output_clean, dim=1))
        loss = criterion(output, target) + loss_advclean + loss_advori
        # print(loss)

                

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        self.model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)

        # measure accuracy
        acc1 = accuracy(output, target, topk=(1,))

        #Logging is done through this module as follows.
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
   

    def on_train_epoch_end(self):
        l2_norm_obj = sum(p.norm(2) for p in model.module.visual.parameters())
        l2_norm_ori = sum(p.norm(2) for p in model_ori.module.visual.parameters())
        ratio = abs(l2_norm_ori - l2_norm_obj) / float(l2_norm_ori)
        abs_l2 = abs(l2_norm_ori - l2_norm_obj)


      
# # def validate(val_loader, texts, model, prompter, add_prompter, criterion, args):
# def validate(val_loader_list, val_dataset_name, texts_list, model, model_text, model_image,
#              prompter, add_prompter, criterion, args):
#     dataset_num = len(val_loader_list)
#     acc_all = []

#     test_stepsize = args.test_stepsize

#     for cnt in range(dataset_num):

#         val_loader = val_loader_list[cnt]
#         texts = texts_list[cnt]
#         dataset_name = val_dataset_name[cnt]

#         batch_time = AverageMeter('Time', ':6.3f')
#         losses = AverageMeter('Loss', ':.4e')
#         top1_org = AverageMeter('Original Acc@1', ':6.2f')
#         top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
#         top1_adv_org = AverageMeter('Adv Original Acc@1', ':6.2f')
#         top1_adv_prompt = AverageMeter('Adv Prompt Acc@1', ':6.2f')

#         progress = ProgressMeter(
#             len(val_loader),
#             [batch_time, losses, top1_org, top1_prompt, top1_adv_org, top1_adv_prompt],
#             prefix=dataset_name + '_Validate: ')

#         # switch to evaluation mode
#         prompter.eval()
#         add_prompter.eval()
#         model.eval()

#         end = time.time()
#         for i, (images, target) in enumerate(tqdm(val_loader)):

#             if 'cifar' not in val_dataset_name:
#                 if i % 20 != 0 and not args.evaluate:
#                     continue

#             images = images.to(device)
#             target = target.to(device)
#             text_tokens = clip.tokenize(texts).to(device)

#             # logger.info(images.size())

#             with autocast():

#                 # clean images, with prompt and without prompt
#                 # compute output
#                 with torch.no_grad():
#                     # prompt_token = add_prompter()
#                     prompt_token = None
#                     # output_prompt, _ = model(prompter(clip_img_preprocessing(images)), text_tokens, prompt_token)
#                     output_prompt= multiGPU_CLIP(model,
#                                                            prompter(clip_img_preprocessing(images)), text_tokens,
#                                                            prompt_token)

#                     loss = criterion(output_prompt, target)

#                     # measure accuracy and record loss
#                     acc1 = accuracy(output_prompt, target, topk=(1,))
#                     losses.update(loss.item(), images.size(0))
#                     top1_prompt.update(acc1[0].item(), images.size(0))

#                     top1_org.update(acc1[0].item(), images.size(0))

#                 torch.cuda.empty_cache()

#                 # generate adv example
#                 if args.CW:
#                     delta_prompt = attack_CW(prompter, model, model_text, model_image, add_prompter, criterion,
#                                              images, target, text_tokens,
#                                              test_stepsize, args.test_numsteps, 'l_inf', epsilon=args.test_eps)
#                 elif args.autoattack:
#                     def model_fn(x):
#                         output_a = multiGPU_CLIP(model,
#                                                           prompter(clip_img_preprocessing(x)),
#                                                           text_tokens,
#                                                           prompt_token)
#                         return output_a.to(torch.float32)

#                     adversary = AutoAttack(model_fn, norm='Linf', eps=args.test_eps, version='standard')
#                     adv_samples = adversary.run_standard_evaluation(images, target, bs=100)
#                     delta_prompt = adv_samples - images
#                     delta_prompt = clamp(delta_prompt, lower_limit - images, upper_limit - images)
#                 else:
#                     delta_prompt = attack_pgd(prompter, model, model_text, model_image, add_prompter, criterion,
#                                               images, target, text_tokens,
#                                               test_stepsize, args.test_numsteps, 'l_inf', epsilon=args.test_eps)

#                 # compute output
#                 torch.cuda.empty_cache()
#                 with torch.no_grad():
#                     prompt_token = add_prompter()
#                     # output_prompt_adv, _ = model(prompter(clip_img_preprocessing(images + delta_prompt)), text_tokens, prompt_token)
#                     output_prompt_adv = multiGPU_CLIP( model,
#                                                                prompter(clip_img_preprocessing(images + delta_prompt)),
#                                                                text_tokens, prompt_token)

#                     loss = criterion(output_prompt_adv, target)

#                 # bl attack
#                 torch.cuda.empty_cache()

#                 # measure accuracy and record loss
#                 acc1 = accuracy(output_prompt_adv, target, topk=(1,))
#                 losses.update(loss.item(), images.size(0))
#                 top1_adv_prompt.update(acc1[0].item(), images.size(0))

#                 acc1 = accuracy(output_prompt_adv, target, topk=(1,))
#                 top1_adv_org.update(acc1[0].item(), images.size(0))

#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()

#         torch.cuda.empty_cache()
     
#         acc_all.append(top1_adv_prompt.avg)

#     return np.mean(acc_all)

      
    def validation_step(self, batch, batch_idx):
      
        input,desired=batch[0],batch[1]
        out=self.forward(input)
        #You could log here the val_loss, or just print something. 
        
    def configure_optimizers(self):
        #Automatically called by PL. So don't worry about calling it yourself. 
        #you'll notice that everything from the init function is stored under the self.hparams object 

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

        #Define scheduler here too if needed. 
        return [optimizer,scheduler]
