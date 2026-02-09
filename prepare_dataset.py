import argparse, os, csv  #os->for file path manipulation and directory trasversal; csv-> for writing output in csv format
from widerface_parser import parse_widerface_count  #import a helper function that parses the Wider face annotation face

def is_image(p):    #Check whether a file path points to an image.
    p = p.lower()   #It converts the path string to lowercase, then checks if it ends with a known image extension. 
    return any(p.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp'])   #Returns true if the file is an image and false otherwise.

def main():
    ap = argparse.ArgumentParser()  #Creates an ArgumentParser for reading command-line options.
    ap.add_argument('--images-root', required=True, help='Root delle immagini per cui costruire il CSV')  #Path to the root folder containing the images.
    ap.add_argument('--ann-file', default=None, help='File wider_face_*_bbx_gt.txt')  #Optional path to the Wider face annotation text file.
    ap.add_argument('--out-csv', required=True, help='CSV di output: path,label_count') #Output csv file where the script will write image paths and face counts.
    ap.add_argument('--no-ann', action='store_true', help='Se usi immagini senza annotazioni') #A flag to tell the script to run without annotations. If this flag is used, the script won't try to read --ann-file. 
    ap.add_argument('--assume-zero', action='store_true', help='Se no-ann, assegna 0 volti a tutte le immagini trovate') #When runing is no-annotation mode, this flag assigns a face count of 0 to all found images.(In the current version this flag always writes 0 anyway.)
    args = ap.parse_args() #Parses all command-line arguments and stores them in the args object.

    rows = []   #Initializes an empty list to collect (image_path, face_count) pairs that will later be writen to the CSV
    if not args.no_ann:
        assert args.ann_file is not None, 'Specifica --ann-file oppure usa --no-ann'  #Checks that an annotation file was provided, otherwise-> error.
        pairs = parse_widerface_count(args.ann_file)  #parse_widerface_count should read the annotation file and return a list of (relative_path, count) tuples.
        for rel, cnt in pairs:    #Loops through each (relatiove_path, count) pair.
            path = os.path.join(args.images_root, rel) #Join the relative path with the root directory to get the absolute image path.
            if os.path.isfile(path):   #If the file actually exists, appends (path, count) to the rows list. This avoids including annotations for missing images. 
                rows.append((path, cnt))  
    else:
        # Folder scan and assign 0 (negativi). Case without anntations.
        for root, _, files in os.walk(args.images_root):  #If no-ann was passed, the script recursively walks through all subfolders of images-root. 
            for f in files:   #For each file found, checks whether it's an image using is_image(). If so, it appends (path, 0)-> meaning that image has zero faces. 
                p = os.path.join(root, f) 
                if is_image(p):
                    rows.append((p, 0 if args.assume_zero else 0)) # 0 if args.assume_zero else 0 ->this always results in 0.

    # Write CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True) #Ensures that the output directory exists before writing the CSV file.
    with open(args.out_csv, 'w', newline='') as f:   #Opens CSV file for writing.
        w = csv.writer(f)                            #Creates a CSV writer object.
        w.writerow(['path', 'count'])                #Writes a header row: path,count.
        for r in rows:         
            w.writerow(r)                            #Then writes each (path, count) tuple from rows to the file.
    print(f'Scritte {len(rows)} righe in {args.out_csv}') #Prints how many row were written and where the CSV was saved. 

if __name__ == '__main__':
    main()


#We create a CSV mapping image paths to the number of faces. We have two models: 1) With annotations: reads face count from WIDER Face annotation file. 2) Without annotations: Scans all images and assigns 0. Output a CSV like :path,count.