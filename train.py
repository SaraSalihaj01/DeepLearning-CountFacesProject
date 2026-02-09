import argparse, os, math, random  #parse command-line arguments, operating system functions, math operations
import numpy as np   #numerical operations on arrays
import torch        #PyTorch
import torch.nn as nn   #Neural network modules
from torch.utils.data import DataLoader, ConcatDataset #Utilities for batching and combining datasets
#from torchvision.ops import sigmoid_focal_loss #Loss function for imbalanced classification problems
from tqdm import tqdm #Progress bar for loops
from data import CountFacesDataset     #Import the custom dataset class and the function to build the main part of the model.
from model import build_model

def set_seed(seed=42):                                                                          
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)   #Fix random seeds to ensures reproducibility (same results each run)      

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))    #Mean Absolute Error (average absolute difference)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2))) # Function to compute Root Mean Squared Error (square error penalizes big mistakes).

def acc_within(y_true, y_pred, tol=0):
    return float(np.mean(np.abs(y_true - y_pred) <= tol))    #Accuracy within a tolerance. tol=0 -> exact match rate ; tol=1 -> within +- faces    

#Sets up the command-line arguments for the script (training CSVs,validation CSVs, model type, epochs, batch size, LR, etc.).                                                            
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-csv', required=True, help='CSV o lista CSV separati da virgola')
    ap.add_argument('--val-csv', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--img-size', type=int, default=256)
    ap.add_argument('--model', default='resnet18')
    ap.add_argument('--scheduler', default='cos', choices=['cos','none'])
    ap.add_argument('--early-stop', type=int, default=10)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)     #Fixes seed for reproducibility
    os.makedirs(args.out_dir, exist_ok=True)  #If it doesn't exist -> creates output directory to save checkpoint/metrics.

    # Datasets and Loaders
    train_csvs = [p.strip() for p in args.train_csv.split(',')]   #Supports multiple training CSVs
    train_ds = ConcatDataset([CountFacesDataset(p, img_size=args.img_size, augment=True) for p in train_csvs]) #Training dataset uses augmentation for robustness.
    val_ds = CountFacesDataset(args.val_csv, img_size=args.img_size, augment=False)  #Validation dataset is clean (no augmentation)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)  #Create Pytorch DataLoaders for batching. Shuffle=True-> doesn't learn the order
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)  #Training is shuffled, validation is not-> deterministic validation  
    
    #Model, Loss and Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Sets device (GPU is available). PyTorch use 'cuda' to indicate GPU.
    model = build_model(args.model, pretrained=True).to(device) #Builds model and moves it to device. ResNet18 by default,pretrained on ImageNet.

    criterion = nn.MSELoss()  #Use MSE loss because we have a regression problem.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  #Optimizer AdamW(Adam with weight decay): adaptive Learning rate method with decoupled weight decay to improve generalization and prevent onverfitting.
    if args.scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None     ##Gradually decreases the learning rate following a cosine curve. Starts with a haigher learning rate to allow fast exploration, then slowly lowers it to fine-tune the model and stabilize training. This helps achieve smoother convergence and better final accuracy.
    
    #Training setup
    best_mae = float('inf')  #Tracks best validation MAE
    patience = args.early_stop #Patience counter for early stopping.
    ckpt_path = os.path.join(args.out_dir, 'best.pt') #Path where best model checkpoint is saved.

    for epoch in range(1, args.epochs+1):  #Iterates through all epochs and batches.For each batch:moves data to the device,performs a forward pass to get predictions,computes the loss,clears previous gradients,performs backpropagation
        model.train()                      #to calculate new ones,applies gradient clipping to prevent instability and updates model weights using the optimizer. The progress tqdm displays the current training loss in real time.
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pbar.set_postfix(loss=float(loss.detach().cpu().item()))
        if scheduler:     
            scheduler.step()   #Each epoch updates learning rate. Learning rate-> how fast the model updates its weight.

        # Validation
        model.eval() #Switch to evaluation mode(no dropout/batch normalization updates). Disable gradients for efficiency.Loop over validation data,get predictions and true counts,convert tensors to Numpy arrays and collect all results for metric computation.
        ys, ps = [], []  #Initialization of two empty lists that will be filled with data from each batch to then calculate the final metrics(MAE, RMSE, etc.)
        with torch.no_grad():   #No gradients. 
            for x, y in val_loader:  #Takes a batch of images(x) and their real value(y) from DataLoader of validation.
                x = x.to(device)  #Moves the images to GPU or CPU. The model and datas should be in the same device.
                pred = model(x).detach().cpu().numpy().reshape(-1) #Calculate the model prediction-> the estimated number of faces in each batch image. From a Pytorch batch tensor we obtain a prediction array ready for metrics.
                y = y.numpy().reshape(-1)
                ys.append(y); ps.append(pred) #Save the real and prediction values. 
        ys = np.concatenate(ys); ps = np.concatenate(ps)  #It combines all the small arrays into one array. We use these two arrays to calculate MAE, RMSE and accuracies.
        m_mae = mae(ys, ps)
        m_rmse = rmse(ys, ps)
        acc0 = acc_within(np.round(ys), np.round(ps), 0)
        acc1 = acc_within(np.round(ys), np.round(ps), 1)
        print(f'Val: MAE={m_mae:.3f} RMSE={m_rmse:.3f} Acc@0={acc0:.3f} Acc@1={acc1:.3f}')

        with open(os.path.join(args.out_dir, 'metrics.csv'), 'a') as f:                #Save metrics per epoch into a CSV log.
            f.write(f'{epoch},{m_mae:.6f},{m_rmse:.6f},{acc0:.6f},{acc1:.6f}\n')
        
        #If validation MAE improves: save model checkpoint, reset patience. Otherwise:decrease patience. If it reaches 0->stop trining early.
        if m_mae < best_mae:
            best_mae = m_mae
            torch.save({'model': model.state_dict(),
                        'config': vars(args)}, ckpt_path)
            patience = args.early_stop
        else:
            patience -= 1
            if patience <= 0:
                print('Early stopping.')
                break
    #Print the best validation MAE achieved.
    print(f'Best MAE: {best_mae:.4f}  (checkpoint: {ckpt_path})')

if __name__ == '__main__':
    main()
