from typing import List, Tuple

def parse_widerface_count(ann_file: str) -> List[Tuple[str, int]]: #Function returns a list of tuples, each containing a string and an integer.
    """This function parses a WIDER Face annotation file (e.g., wider_face_train_bbx_gt_txt) and returns a list of pairs.
    Each pair contains the relative path of an image and the number of valid face bounding boxes it contains (where both width (w>0) and height (h>0) are greater than zero).
    """
    pairs = []   #initializes an empty list that will store all (image_path,count) pairs.
    with open(ann_file, 'r') as f:   #Open the annotation file for reading.
        lines = [ln.strip() for ln in f.readlines() if ln.strip() != '']  #Reads all lines, strips whitespace from each, and removes empty lines. The resulting list lines contains all non-empty lines of text.
    i = 0     #Inizializes an index variable i to manually iterate through the list. 
    n = len(lines)  #Stores the total number o f lines in n.
    while i < n:
        img_rel = lines[i]     # The first line of each blocks represents the relative image path, eg: 0--Parade/0_Parade_marchingband_1_849.jpg
        i += 1     #Moves to the next line.
        if i >= n:   #Safety check: if the file ends unexpectedly after the image name, exit the loop.
            break   
        try:
            num = int(lines[i])   #The next line should contain the number of bounding boxes for that image. Converts it to an integer, if this fails it breaks the loop to prevent errors.
        except ValueError:
            break
        i += 1   #Moves to the next line, where bounding box coordinates start.
        count = 0  #Inizializes a local counter to count the valid boxes.
        for _ in range(num):   #Iterates through the next num lines, each representing one bounding box.
            if i >= n:
                break
            parts = lines[i].split()  #Splits each lines into components(usually x,y,w,h, expression, blur...)
            i += 1                   #Moves the index forwards by one each iteration.
            if len(parts) >= 4:      #The line contains at least four values (x,y,w,h).
                w = int(float(parts[2]))  #Converts width and height to integers
                h = int(float(parts[3]))
                if w > 0 and h > 0:   #If both are positive, increments the face counter.
                    count += 1
        pairs.append((img_rel, count))   #After processing all boxes for one image, appends a tuple (relative_image_path, face_count) to the results list.
    return pairs  #Returns the complete list of (path,count) pairs for all images in the fail.
