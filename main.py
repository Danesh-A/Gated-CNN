import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF 
from tqdm import tqdm
from models.model import Unet, GscnnUnet4Gates, GscnnUnet5Gates
from datasets.dataset import ThreeClassHE
from utils.losses import DiceLoss
from utils.metrics import DiceMetric, overlapmetricdice, sensitivity, specificity
import seaborn as sns
import yaml
from torch.utils.data import DataLoader

def seed_everything(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
seed_everything(42)

def load_config(path="config.yml"):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config



sns.set()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
  
    
def fit(epochs,model,train_loader,val_loader,test_loader,optimizer,runname,mode,all_dir, save_images = False):
    torch.cuda.empty_cache()
    
    trainwriter = SummaryWriter(f'{all_dir}tb/{runname}/train')
    validwriter = SummaryWriter(f'{all_dir}tb/{runname}/valid')

    bceloss = nn.BCELoss().to(device)
    diceloss = DiceLoss()
    
    train_losses = []
    val_losses = []    
    train_dice = []
    val_dice = []
    
    model.to(device)
    print("Beginning Training:")
    for e in range(epochs):        
        #Training Metrics
        running_loss = 0
        running_dice_loss = 0
        
        running_dice = 0
        running_edge_dice = 0
        running_nuclei_dice = 0 
      
        running_edgebce_loss = 0

###############################################################################
        #Training Loop
        model.train()
        

        for i, data in enumerate(tqdm(train_loader)):
            image = data["image"]
            mask = data["mask"]
            wm = data["wm"]
            images = image.to(device)
            masks = mask.to(device)
            wm = wm.to(device)

            if mode == 'gated':
                gradient = data["gradient"]
                gradmask = data["gradmask"]
                gradient = gradient.to(device)
                gradmask = gradmask.to(device)
                               
            if mode == 'gated':
                output, shapeout = model(images.float(),gradient.float())
                
                #LOSSES
                train_dice_loss = diceloss(output, masks,'a','softmax')
                edgebceloss = bceloss(shapeout,gradmask)  
                loss = train_dice_loss + 5 * edgebceloss 
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                #METRICS
                dice = DiceMetric(output, masks,'a','softmax')
                edgedice = DiceMetric(output, masks,'e','softmax')
                nucleidice  = DiceMetric(output, masks,'n','softmax')
                

            else:
                output = model(images.float())
                
                #LOSSES
                train_dice_loss =diceloss(output, masks,'a','softmax')
                loss = train_dice_loss 
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                #METRICS
                dice = DiceMetric(output, masks,'a','softmax')
                edgedice = DiceMetric(output, masks,'e','softmax')
                nucleidice  = DiceMetric(output, masks,'n','softmax')   
            
            #EPOCH SUMMATION
            if mode == 'gated':  
                #LOSSES
                running_loss += loss.item()
                running_dice_loss += train_dice_loss.item()
                running_edgebce_loss += edgebceloss.item()
 
                #METRICS
                running_dice += dice.item()
                running_edge_dice +=edgedice.item()
                running_nuclei_dice += nucleidice.item()
            else:
                #LOSSES
                running_loss += loss.item()
                running_dice_loss += train_dice_loss.item()
                #METRICS
                running_dice += dice.item()
                running_edge_dice +=edgedice.item()
                running_nuclei_dice += nucleidice.item()
                
        if mode == 'gated':    
            train_dice.append(running_dice/len(train_loader))
            train_losses.append(running_loss/len(train_loader))
            
            trainwriter.add_scalar('Losses/Total Loss',running_loss/len(train_loader),e)
            trainwriter.add_scalar('Losses/Dice Loss',running_dice_loss/len(train_loader),e)
            trainwriter.add_scalar('Losses/Edge BCE Loss',running_edgebce_loss/len(train_loader),e)
 
            trainwriter.add_scalar('Metrics/Overall Dice',running_dice/len(train_loader),e)
            trainwriter.add_scalar('Metrics/Edge Dice',running_edge_dice/len(train_loader),e)
            trainwriter.add_scalar('Metrics/Nuclei Dice',running_nuclei_dice/len(train_loader),e)
        else:
            trainwriter.add_scalar('Losses/Total Loss',running_loss/len(train_loader),e)
            trainwriter.add_scalar('Losses/Dice Loss',running_dice_loss/len(train_loader),e)

            trainwriter.add_scalar('Metrics/Overall Dice',running_dice/len(train_loader),e)
            trainwriter.add_scalar('Metrics/Edge Dice',running_edge_dice/len(train_loader),e)
            trainwriter.add_scalar('Metrics/Nuclei Dice',running_nuclei_dice/len(train_loader),e)

        print(f"Epoch: {e+1} Training Loss: {running_loss/len(train_loader)}")
        
        model.eval()
        
        validation_loss = 0
        validation_dice_loss = 0
        validation_edgebce_loss = 0
        
        validation_dice = 0
        validation_edge_dice = 0
        validation_nuclei_dice = 0

        val_test_loss = 0

        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                image = data["image"]
                mask = data["mask"]
                wm = data["wm"]
                images = image.to(device)
                masks = mask.to(device)
                wm = wm.to(device)
                
                if mode == 'gated':
                    gradient = data["gradient"]
                    gradmask = data["gradmask"]
                    
                    gradient = gradient.to(device)
                    gradmask = gradmask.to(device)
                
                if mode == 'gated':
                    output, shapeout = model(images.float(),gradient.float())
                    #LOSSES
                    valid_dice_loss = diceloss(output, masks, 'a', 'softmax')
                    edgebceloss =  bceloss(shapeout,gradmask) 
                    loss = valid_dice_loss + 5* edgebceloss 
                    
                    #METRICS
                    dice = DiceMetric(output, masks,'a','softmax')
                    edgedice = DiceMetric(output, masks,'e','softmax')
                    nucleidice  = DiceMetric(output, masks,'n','softmax')
                

                else:
                    output = model(images.float())
                    
                    valid_dice_loss = diceloss(output,masks,'a','softmax')
                    loss = valid_dice_loss
                    
                    dice = DiceMetric(output, masks,'a','softmax')
                    edgedice = DiceMetric(output, masks,'e','softmax')
                    nucleidice  = DiceMetric(output, masks,'n','softmax')
                
                if mode == 'gated':
             
                    validation_loss += loss.item()
                    validation_dice_loss += valid_dice_loss.item()
                    validation_edgebce_loss += edgebceloss.item()

                    
                    validation_dice += dice.item()
                    validation_edge_dice +=edgedice.item()
                    validation_nuclei_dice += nucleidice.item()
                else:
                    #LOSSES
                    validation_loss += loss.item()
                    validation_dice_loss += valid_dice_loss.item()
                    #METRICS
                    validation_dice += dice.item()
                    validation_edge_dice +=edgedice.item()
                    validation_nuclei_dice += nucleidice.item()
                    
        val_dice.append(validation_dice/len(train_loader))        
        val_losses.append(validation_loss/len(val_loader))
        
        if mode == 'gated':
            validwriter.add_scalar('Losses/Total Loss',validation_loss/len(val_loader),e)
            validwriter.add_scalar('Losses/Dice Loss',validation_dice_loss/len(val_loader),e)
            validwriter.add_scalar('Losses/Edge BCE Loss',validation_edgebce_loss/len(val_loader),e)

            validwriter.add_scalar('Metrics/Overall Dice',validation_dice/len(val_loader),e)
            validwriter.add_scalar('Metrics/Edge Dice',validation_edge_dice/len(val_loader),e)
            validwriter.add_scalar('Metrics/Nuclei Dice',validation_nuclei_dice/len(val_loader),e)
        else:
            validwriter.add_scalar('Losses/Dice Loss',validation_dice_loss/len(val_loader),e)
            validwriter.add_scalar('Losses/Total Loss',validation_loss/len(val_loader),e)

            validwriter.add_scalar('Metrics/Overall Dice',validation_dice/len(val_loader),e)
            validwriter.add_scalar('Metrics/Edge Dice',validation_edge_dice/len(val_loader),e)
            validwriter.add_scalar('Metrics/Nuclei Dice',validation_nuclei_dice/len(val_loader),e)
        
        
        print(f"Epoch: {e+1} Validation Loss: {validation_loss/len(val_loader)}")
        
###############################################################################
        #Testing
    model.eval()
    dices = []
    edgedices = []
    nucleidices = []
    dices15 = []
    edgedices15 = []
    nucleidices15 = []
    edgesens = []
    edgespec = []
    nucsens = []
    nucspec = []
    
    if save_images:
        savedir = f'{all_dir}image_results/{runname}/tupac_results/'
        os.makedirs(savedir,exist_ok=True)
    
    modeldir = f'{all_dir}models/'+ runname + '.pth'    


    torch.save(model.state_dict(),modeldir)    
    
    for i,data in enumerate(tqdm(test_loader)):
        image = data["image"]
        mask = data["mask"]
        wm = data["wm"]
        images = image.to(device)
        masks = mask.to(device)
        wm = wm.to(device)
        if mode == 'gated':
            gradient = data["gradient"]
            gradmask = data["gradmask"]
            gradient = gradient.to(device)
            gradmask = gradmask.to(device)
        
        if mode == 'gated':
            output, shapeout = model(images.float(),gradient.float())
            tmpdsc = DiceMetric(output,masks,'a','softmax').item()
            tmpdsc2 = DiceMetric(output,masks,'e','softmax').item()
            tmpdsc3= DiceMetric(output,masks,'n','softmax').item()
            dice15, nuclei15, edge15 = overlapmetricdice(output, masks,wm,0.15)

            output = torch.softmax(output,dim = 1)
            shapeout = TF.to_pil_image(shapeout[0].cpu())
            
            if save_images:
                shape_path = os.path.join(savedir, f"shape_{i}.png")
                shapeout.save(shape_path)
       
        else:
            output = model(images.float())
            tmpdsc = DiceMetric(output,masks,'a','softmax').item()
            tmpdsc2 =  DiceMetric(output,masks,'e','softmax').item()
            tmpdsc3= DiceMetric(output,masks,'n','softmax').item()
            dice15, nuclei15, edge15 = overlapmetricdice(output, masks,wm,0.15)
            output = F.softmax(output, dim = 1)
            
        edgesens.append(sensitivity(output, masks, 1))
        edgespec.append(specificity(output, masks, 1))
        nucsens.append(sensitivity(output, masks, 0))
        nucspec.append(specificity(output, masks, 0))
        
        dices15.append(dice15.item())
        edgedices15.append(edge15.item())
        nucleidices15.append(nuclei15.item())        
        
        images = TF.to_pil_image(images[0].cpu())
        masks = TF.to_pil_image(masks[0].cpu())
        output = TF.to_pil_image(output[0].cpu()) 
           
        
        dices.append(tmpdsc)
        edgedices.append(tmpdsc2)
        nucleidices.append(tmpdsc3)
        
        
        if save_images:
            images_path = os.path.join(savedir, f"input_{i}.png")
            images.save(images_path)
        
            masks_path = os.path.join(savedir, f"mask_{i}.png")
            masks.save(masks_path)
        
            output_path = os.path.join(savedir, f"output_{i}.png")
            output.save(output_path)  
      
if __name__ ==  '__main__':
    

    config = load_config()
    root_dir = config['data']['root_dir']
    dataset_name = config['data']['dataset_name']
    norm_method = config['data']['normalization']
    mode = "train"

    train_dataset = ThreeClassHE(root_dir,dataset_name,norm_method,mode,transform =  None, all_transform = None)
    valid_dataset = ThreeClassHE(root_dir,dataset_name,norm_method,"valid",transform = None, all_transform = None)
    test_dataset = ThreeClassHE(root_dir,dataset_name,norm_method,"test",transform = None, all_transform = None)


    train_loader = DataLoader(train_dataset, batch_size = config['data']['batch_size'],shuffle = config['data']['shuffle'], pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size = config['data']['batch_size'],shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = 1,shuffle = False)

    Model_Map = {
    'UNet': Unet,
    'GscnnUnet4Gates': GscnnUnet4Gates,
    'GscnnUnet5Gates': GscnnUnet5Gates
    }
    
    model_name = config['model']['architecture']
    model = Model_Map[model_name]((config['model']['input_channels']),(config['model']['output_channels']))
    model = model.to(device)
    mode = 'gated'
    
    lr = config['training']['lr']
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = config['training']['epochs']
    name = config['logging']['run_name']

    save_dir = config['logging']['log_dir']

    fit(epochs,model,train_loader,valid_loader,test_loader, optimizer,scheduler,name,mode,save_dir,save_images = True)