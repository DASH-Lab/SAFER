import torch
from torch import nn
from torch import optim

from torchvision.models.feature_extraction import create_feature_extractor

import time
from util import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def idx2onehot(idx, n):
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)

    return onehot

def get_model_variables(net, model):
    if model == "resnet18":
        feat_extractor = create_feature_extractor(net, return_nodes={'avgpool':'feat'})
        classifier = net.fc
        input_dim = 512
    elif model == "resnet50":
        feat_extractor = create_feature_extractor(net, return_nodes={'avgpool':'feat'})
        classifier = net.fc
        input_dim = 2048
    elif model == "vit":
        feat_extractor = create_feature_extractor(net, return_nodes={'getitem_5':'feat'})
        classifier = net.heads.head
        input_dim = 768
    return feat_extractor, classifier, input_dim


class SAFER(nn.Module):
    def __init__(self, model, model_name, num_class, latent_size=4):
        super(SAFER, self).__init__()
        #assert(model_name == "resnet18")
        feat_extractor, classifier, input_dim = get_model_variables(model, model_name)

        self.num_class = num_class
        self.input_dim = input_dim          # can be different values depending on model
        self.latent_size = latent_size      # can be different values depending on model

        self.linear_means = nn.Linear(input_dim+num_class, latent_size)       # needs as much as # of class
        self.linear_log_var = nn.Linear(input_dim+num_class, latent_size)
        
        self.mlp = feat_extractor
        self.fc = classifier

        self.dec = nn.Linear(latent_size+num_class, input_dim) 

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + std * eps # Force this become normal distriution by KL loss
    
    def encode(self, x, c):
        if x.dim() > 2:
            x = x.view(-1, self.input_dim)

        c = idx2onehot(c, n=self.num_class)
        x = torch.cat((x, c), dim=1)        # x = feature dimension + dim(c)

        means = self.linear_means(x)        # in: feature dimension + dim(c) / out: latent_size=4
        log_vars = self.linear_log_var(x)   # in: feature dimension + dim(c) / out: latent_size=4

        return means, log_vars
    
    def decode(self, z, c):
        c = idx2onehot(c, n=self.num_class)
        z = torch.cat((z, c), dim=-1)

        x = self.dec(z) # in: latent_size=4 + dim(c) / out: feature dimension
        return x
    
    def forward(self, x, c=None):
        x = self.mlp(x) # out: feature dimension
        x = torch.flatten(x['feat'], start_dim=1)
            
        means, log_vars = self.encode(x, c)
        z = self.reparameterize(means, log_vars)
        recon_x = self.decode(z, c)    # in: latent_size=4 + dim(c) / out: feature dimension

        x = (x + recon_x) / 2.0        
        out = self.fc(x)
        
        return out, x, recon_x, means, log_vars

def reg_loss_fn(criterion, outputs, labels, recon_x, x, mean, log_var, global_mean, args):
    superloss = nn.MSELoss()

    CE = criterion(outputs, labels) 
    recon_loss = superloss(recon_x, x.detach())
    KLD = (-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())) / x.size(0)

    gm = global_mean.detach()
    dist = torch.norm(mean - gm, dim=1)    # distance between mean and global_mean
    dist_mean = dist.mean().clamp_min(1e-6)
    spread_loss = 1.0 / dist_mean
    
    if args.ablation_option == 0 or args.ablation_option == 3: 
        total = CE + KLD + recon_loss + 0.5*spread_loss
    elif args.ablation_option == 2 or args.ablation_option == 4: 
        total = CE + KLD + recon_loss 
    elif args.ablation_option == 1 or args.ablation_option == 5: 
        total = CE + KLD + 0.5*spread_loss

    return total


def unlearning_by_safer(model, data_loader_dict, args, logger):
    forget_loader = data_loader_dict['forget']
    retain_loader = data_loader_dict['retain']
    
    safer = SAFER(model, args.model, num_class=args.num_class, latent_size=4).to(DEVICE)
    
    epochs = args.epochs #10
    criterion = get_criterion(args)
    optimizer = optim.SGD(safer.parameters(), lr=args.learning_rate , momentum=0.9, weight_decay=5e-4)

    alpha = 0.1
    global_mean = None
  
    forgot_targets = list(dict.fromkeys(args.forgot_indices))
    forgot_tensor = torch.tensor(forgot_targets, dtype=torch.int64).cuda() if forgot_targets else torch.empty(0, dtype=torch.int64).cuda()      
    
    for epoch in range(epochs):
        safer.train()
        start = time.time()

        if args.ablation_option < 3: 
            for data in forget_loader:

                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)              
                x = safer.mlp(inputs)
                x = torch.flatten(x['feat'], start_dim=1)
                out = safer.fc(x)
                
                random_targets = torch.rand_like(out)
                
                unique_labels = labels.unique()
                f_labels = torch.cat([unique_labels, forgot_tensor])
                random_targets[:, f_labels] = 0.        # if class-aligned case
                    
                random_targets = random_targets / random_targets.sum(dim=1, keepdim=True)
                loss = criterion(out, random_targets)

                total_loss = loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()   

        for data in retain_loader:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs, x, recon_x, means, log_vars = safer(inputs, labels)

            batch_mean = means.mean(dim=0)
            if global_mean is None:
                global_mean = batch_mean.detach()
            else:
                global_mean = (1 - alpha) * global_mean + alpha * batch_mean
            
            loss = reg_loss_fn(criterion, outputs, labels, recon_x, x, means, log_vars, global_mean, args)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        args.runtime += time.time()-start
        
        safer.eval()
        
        print_results(model, data_loader_dict, epoch, args, logger)

    return model
