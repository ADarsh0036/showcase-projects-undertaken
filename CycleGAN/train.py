import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import HorseZebraDataset
import utils
import config
from utils import save_checkpoint
from generator import Generator
from discriminator import Discriminator
from tqdm import tqdm
import os


## G:X(Horse) -> Y(Zebra)   Dx discriminates for horse
## F:Y(Zebra) -> X(Horse)   Dy discriminates for Zebra

def train_fn(G, F, Dx, Dy, mse, L1, g_scalar, d_scalar, disc_opt, gen_opt, lambda_cyc, loader):
    
    loop = tqdm(loader, leave = True)
    
    for batch_idx, (horse, zebra) in enumerate(loop):
        horse = horse.to(config.DEVICE)
        zebra = zebra.to(config.DEVICE)
        #Train Discriminator:
        with torch.cuda.amp.autocast():
            # X -> Y
            fake_zebra = G(horse)
            real_zebra_prob = Dy(zebra)
            fake_zebra_prob = Dy(fake_zebra.detach())
            D_Z_loss = mse(real_zebra_prob, torch.ones_like(real_zebra_prob)) + mse(fake_zebra_prob, torch.zeros_like(fake_zebra_prob))
            # Y -> X
            
            fake_horse = F(zebra)
            real_horse_prob = Dx(horse)
            fake_horse_prob = Dx(fake_horse.detach())
            D_H_loss = mse(real_horse_prob, torch.ones_like(real_horse_prob)) + mse(fake_horse_prob, torch.zeros_like(fake_horse_prob))
            
            D_loss = (D_Z_loss + D_H_loss) / 2
            
        disc_opt.zero_grad()
        d_scalar.scale(D_loss).backward()
        d_scalar.step(disc_opt)
        d_scalar.update()
        
        #Train Generator
        
        with torch.cuda.amp.autocast():
            
            
            fake_zebra_prob = Dy(fake_zebra)
            fake_horse_prob = Dx(fake_horse)
            G_Z_loss = mse(fake_zebra_prob, torch.ones_like(fake_zebra_prob))
            G_H_loss = mse(fake_horse_prob, torch.ones_like(fake_horse_prob))
            
            #Cycle Loss:
            
            Lcyc_Horse = L1(horse, F(fake_zebra))
            Lcyc_Zebra = L1(zebra, G(fake_horse))
            
            G_loss = G_Z_loss + G_H_loss + lambda_cyc * Lcyc_Horse + lambda_cyc * Lcyc_Zebra
            
            
        gen_opt.zero_grad()
        g_scalar.scale(G_loss).backward()
        g_scalar.step(gen_opt)
        g_scalar.update()
        
        
        if batch_idx %  200 == 0:
            
            if not os.path.exists(config.TRAIN_DIR + "saved_images2"):
                os.mkdir(config.TRAIN_DIR + "saved_images2")
                
            save_image(horse * 0.5 + 0.5, f"saved_images2/horse_{batch_idx}.jpg")
            save_image(fake_horse * 0.5 + 0.5, f"saved_images2/fake_horse_{batch_idx}.jpg")
            save_image(zebra * 0.5 + 0.5, f"saved_images2/zebra_{batch_idx}.jpg")
            save_image(fake_zebra * 0.5 + 0.5, f"saved_images2/fake_zebra_{batch_idx}.jpg")
            
        
def main():
    
    G = Generator(in_channels = 3).to(config.DEVICE)
    F = Generator(in_channels = 3).to(config.DEVICE)
    Dx = Discriminator().to(config.DEVICE)
    Dy = Discriminator().to(config.DEVICE)
    
    gen_opt = optim.Adam(list(G.parameters()) + list(F.parameters()), lr = config.LEARNING_RATE, betas = (0.5, 0.999))
    disc_opt = optim.Adam(list(Dx.parameters()) + list(Dy.parameters()), lr = config.LEARNING_RATE, betas = (0.5, 0.999))
    
    train_dataset = HorseZebraDataset(root_zebra=config.TRAIN_DIR+"/trainB",root_horse=config.TRAIN_DIR+"/trainA",transform=config.transforms)
    train_loader = DataLoader(train_dataset,batch_size=config.BATCH_SIZE,pin_memory=True,shuffle=True,num_workers=config.NUM_WORKERS)
    val_dataset = HorseZebraDataset(root_zebra=config.VAL_DIR+"/testB",root_horse=config.VAL_DIR+"/testA",transform=config.transforms)
    val_loader = DataLoader(val_dataset,batch_size= 1 ,pin_memory=True,shuffle=False,num_workers=config.NUM_WORKERS)
    
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_H,gen_H,gen_opt,config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_Z,gen_Z,gen_opt,config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_H,disc_H,disc_opt,config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_Z,disc_Z,disc_opt,config.LEARNING_RATE)

    g_scalar = torch.cuda.amp.GradScaler()
    d_scalar = torch.cuda.amp.GradScaler()
    
    mse = nn.MSELoss()
    L1 = nn.L1Loss()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(G,F,Dx,Dy, mse, L1, g_scalar, d_scalar, disc_opt,gen_opt, config.LAMBDA_CYCLE, train_loader)
        if config.SAVE_MODEL:
            save_checkpoint(G, gen_opt, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(F, gen_opt, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(Dx, disc_opt, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(Dy, disc_opt, filename=config.CHECKPOINT_CRITIC_Z)

if __name__=="__main__":
    main()