import os
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


from tqdm import tqdm
from unet import UNet
from Seismic_Analysis_Pytorch.evaluate import evaluate
from torch import optim
from torch.utils.data import DataLoader
from utils.dice_score import dice_loss
from utils.datapreparation import my_division_data, article_division_data
from utils.data_loading import CustomDataset

def train_model(
        model,
        device,
        article_split: bool = False,
        use_validation: bool = True,
        amp: bool = True,
        batch_size: int = 16,
        n_epochs: int = 50,
        model_name: str = 'test',
        slice_shape1: int = 992,
        slice_shape2: int = 64
):

    
    # 1. Create Dataset 
    transform = transforms.Compose([transforms.ToTensor()])
    if(article_split==False):
        train_image, train_label, test_image, test_label, val_image, val_label=my_division_data(shape=(slice_shape1,slice_shape2), stridetrain=(230,64), strideval=(230,64), stridetest=(230,64))
    else:
        train_image, train_label, val_image, val_label=article_division_data(shape=(slice_shape1,slice_shape2), stridetrain=(230,64), strideval=(230,64))
    val_set=CustomDataset(val_image, val_label, transform)
    train_set=CustomDataset(train_image, train_label, transform)
    n_val = int(len(val_set))
    n_train = len(train_set)
    
    # 2. Create Data LOaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 3. Set up optimizer, loss
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step=0


    logging.info(f'''Starting training:
        Epochs:          {n_epochs}
        Batch size:      {batch_size}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        Mixed Precision: {amp}
        Using Validation: {use_validation}
        Slice Shape: {slice_shape1, slice_shape2}
        Save file: '/scratch/nuneslima/models/{model_name}.pth'
    ''')

    # 4. Begin Training
    for epoch in range(1, n_epochs + 1):
        #Training Round
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{n_epochs}', unit='img') as pbar:
            for batch in train_loader:
                data, target = batch
                data = data.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                target = target.to(device=device, dtype=torch.long)
                
                with torch.autocast(device.type # if device.type != 'mps' else 'cpu'
                                    , enabled=amp):
                    masks_pred = model(data)
                    loss = criterion(masks_pred, target)
                    loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(target, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                    )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(data.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
        
        val_score = evaluate(model, val_loader, device, amp)
        scheduler.step(val_score)
        logging.info('Validation Dice score: {}'.format(val_score))
    torch.save(model, '/scratch/nuneslima/models/'+ model_name +'.pth')




def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--name', '-f', type=str, default=False, help='Model name for saving')
    parser.add_argument('--split', action='store_true', default=False, help='Use article data split')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--use_validation', action='store_true', default=False, help='Use validation on training')
    parser.add_argument('--slice_shape1', '-s1',dest='slice_shape1', metavar='S', type=int,default=992, help='Shape 1 of the slices used in training and validation')
    parser.add_argument('--slice_shape2', '-s2',dest='slice_shape2', metavar='S', type=int,default=64, help='Shape 2 of the slices used in training and validation')
    return parser.parse_args()



if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=1, n_classes=6, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    model.to(device=device)
    try:
        train_model(
            model=model,
            device=device,
            article_split=args.split,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            amp=args.amp,
            model_name=args.name,
            slice_shape1=args.slice_shape1,
            slice_shape2=args.slice_shape2
        )
    except torch.cuda.OutOfMemoryError:
        logging.info('Detected OutOfMemoryError! '
                    'Enabling checkpointing to reduce memory usage, but this slows down training. '
                    'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()