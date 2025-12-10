!pip install opencv-python pandas tqdm
from google.colab import drive
drive.mount('/content/drive')
import os, cv2, numpy as np, pandas as pd
DATASET_RAW = "/content/drive/MyDrive/AI_TraceFinder/Wikipedia"
OUTPUT_ROOT = "/content/drive/MyDrive/AI_TraceFinder_preprocessed/Wikipedia"
OUTPUT_CSV = os.path.join(OUTPUT_ROOT, "wikipedia_labels.csv")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
records=[]

for scanner_model in os.listdir(DATASET_RAW):
    scanner_path = os.path.join(DATASET_RAW, scanner_model)
    if not os.path.isdir(scanner_path): continue
    for root, dirs, files in os.walk(scanner_path):
        for f in files:
            if os.path.splitext(f)[1].lower() not in VALID_EXT: continue
            in_path=os.path.join(root,f)
            rel=os.path.relpath(in_path, DATASET_RAW)
            out_path=os.path.join(OUTPUT_ROOT, os.path.splitext(rel)[0]+SAVE_FORMAT)

            # Read & preprocess
            img = cv2.imdecode(np.fromfile(in_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None: continue
            if img.ndim==3 and img.shape[2]==4: img=img[:,:,:3]
            img=cv2.resize(img, TARGET_SIZE)
            if CONVERT_GRAY and img.ndim==3: img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imencode(SAVE_FORMAT,img)[1].tofile(out_path)

            records.append({"File_name": os.path.basename(out_path),
                            "Dataset":"Wikipedia",
                            "Scanner_Model":scanner_model,
                            "Width":TARGET_SIZE[0],
                            "Height":TARGET_SIZE[1],
                            "Format":SAVE_FORMAT.replace(".","").upper(),
                            "Path":out_path})

df_wiki=pd.DataFrame(records)
df_wiki.to_csv(OUTPUT_CSV,index=False)
print("Wikipedia preprocessing done. Images:", len(df_wiki))
