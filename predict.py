import argparse, os, glob, torch, pandas as pd  #argparse->handles command-line arguments; os,glob->filesystem operations; torch->loads the trained model and runs inference; pandas->saves results(predictions) into a CSV file
from PIL import Image     #opens images
from torchvision import transforms  #preprocessing for images
from model import build_model  #custom function to construct the face-counting model

def is_image(p):
    return p.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))  #Checks if a file path ends with a valid image extension. Non image files are skipped.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True) #path to the trained model checkpoint
    ap.add_argument('--images', required=True, help='Cartella di immagini') #path to a folder containing images to process
    ap.add_argument('--img-size', type=int, default=256)     #resize images to this size before inference
    ap.add_argument('--out-csv', default='preds.csv')      #where to save predictions      
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')   #loads the checkpoint file(.pt or .pth)
    cfg = ckpt.get('config', {})     #reads its config (to know which model was used)
    model = build_model(cfg.get('model','resnet18'), pretrained=False)  #builds the model
    model.load_state_dict(ckpt['model'])  #loads the trained weights into the model
    model.eval()         #sets the model to evaluation mode

#Defines preprocessing for each image:
    t = transforms.Compose([                     
        transforms.Resize((args.img_size, args.img_size)), #resize to (img_size, img_size)
        transforms.ToTensor(),      #convert image to PyTorch tensor
        transforms.ConvertImageDtype(torch.float32),   #convert to float32 (standard for inference)
    ])

    rows = []    #will store predictions
    with torch.no_grad():    #disables gradient tracking (faster inference)
        for root, _, files in os.walk(args.images):  #walks through all files in the given folder (and subfolders)
            for f in files:   #for each file, skip if it's not an image
                p = os.path.join(root, f)
                if not is_image(p): continue
                img = Image.open(p).convert('RGB')   #open and convert to RGB
                x = t(img).unsqueeze(0)    #apply preprocessing(t); unsqueeze(0)->adds a batch dimension (so the model sees a batch of size 1) 
                y = model(x).squeeze().item()  #run the model-> y =model(x) ->prediction; squeeze().item()->turn the tensor into a plain Python number
                rows.append((p, max(0, round(y))))  #round to nearest integer, but ensure it's not negative (can't have -1 faces). Append (path, predicted_count) to the list.

    df = pd.DataFrame(rows, columns=['path','pred_count'])   #convert the results list into a Pandas DataFrame with 2 columns: image file path and predicted number of faces.
    df.to_csv(args.out_csv, index=False)  #save predictions to CSV
    print(f'Salvato {args.out_csv} con {len(df)} righe.') #print a confirmation message with the number of rows.

if __name__ == '__main__':
    main()
