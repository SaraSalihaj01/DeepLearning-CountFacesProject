import argparse, torch, numpy as np, pandas as pd   #argparse-> used to analyse command-line arguments
from torch.utils.data import DataLoader        #loads dataset batches for evaluation
from data import CountFacesDataset
from model import build_model               #function that builds the CNN model

def mae(y_true, y_pred): return np.mean(np.abs(y_true - y_pred))  #Mean Absolute Error: average of absolute differences between true and predicted counts
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred)**2)))  #Root Mean Squared Error:penalizes larger errors more heavily.
def acc_within(y_true, y_pred, tol=0): return float(np.mean(np.abs(y_true - y_pred) <= tol)) #Accuracy within tolerance: tol=0->counts how often the prediction is exactly correct.tol=1->counts predictions that are at most +-1 error.

def main():
    ap = argparse.ArgumentParser()      #Creates an argument parser with options.
    ap.add_argument('--ckpt', required=True)    #path to the model checkpoint file
    ap.add_argument('--val-csv', required=True) #Csv file with validation images and labels
    ap.add_argument('--batch-size', type=int, default=64) #batch size for evaluation
    ap.add_argument('--img-size', type=int, default=256)  #input image size
    args = ap.parse_args()
    
    #Upload the checkpoint saved during training.
    ckpt = torch.load(args.ckpt, map_location='cpu')    #loads the checkpoint file (.pt or .pth)
    cfg = ckpt.get('config', {})                      #if available->extracts the training configuration  
    model = build_model(cfg.get('model','resnet18'), pretrained=False)  #builds the model (default=ResNet18)
    model.load_state_dict(ckpt['model'])                       #loads the trained weights (state_dict) from checkpoint
    model.eval()     #Sets model to evaluation mode

    ds = CountFacesDataset(args.val_csv, img_size=args.img_size, augment=False)  #Creates a dataset from validation csv and wraps it in a DataLoader for batch iteration.
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)   #shuffle= False-> the order doesn't influent the metrics-> deterministic.

    ys, ps = [], []          #initialize empty lists for ground truth(ys) and predictions(ps)      
    with torch.no_grad():    #disables gradient tracking to have a faster and memory efficient inference
        for x, y in dl:                  
            pred = model(x).detach().cpu().numpy().reshape(-1)  # model(x)-> get predictions for the batch; .detach()...->convert predictions to flat NumpPy array. NumPy arrays can only live in cpu, so we need to copy the tensor from GPU to CPU (because the models during training may run in GPU).
            y = y.numpy().reshape(-1)      #convert true labels to flat NumPy array. (Each prediction correspond to a single number (face count) not a nested array.)
            ys.append(y); ps.append(pred)   #append both to lists
    ys = np.concatenate(ys); ps = np.concatenate(ps)  #after loop, concatenate lists into full arrays of all predictions and labels 
  
  #Compute evaluation matrics: 
    m_mae = mae(ys, ps)    #Mean Absolute Error      
    m_rmse = rmse(ys, ps)  #Root Mean Squared Error
    acc0 = acc_within(np.round(ys), np.round(ps), 0)   #Accuracy of exact predictions(after rounding)
    acc1 = acc_within(np.round(ys), np.round(ps), 1)   #Accuracy within +-1 faces (after arrounding)

    print(f'MAE: {m_mae:.4f}\nRMSE: {m_rmse:.4f}\nAcc@0: {acc0:.4f}\nAcc@1: {acc1:.4f}')

if __name__ == '__main__':    #if the script is executed directly, main() function runs.
    main()


#This script loads a trained face-counting model, evaluates it on a validation set, computes standard regression metrics(MAE, RMSE) and reports accuracy of near-exact predictions.
#Metrics.csv is writen during training epoch by epoch and eval.py print the final metric in validation.