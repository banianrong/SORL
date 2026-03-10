import numpy as np
from PIL import Image
import os

def convert_from_ADE(file_lab_in, file_lab_out, map_file='mapFromADE.txt'):
    index_mapping = np.loadtxt(map_file, dtype=np.int32)
    lab = np.array(Image.open(file_lab_in).convert('RGB'))
    labADE = (lab[:, :, 0].astype(np.uint16) // 10) * 256 + lab[:, :, 1].astype(np.uint16)
    labOut = np.zeros_like(labADE, dtype=np.uint8)
    for row in index_mapping:
        challenge_id, ade_id = row
        mask = labADE == ade_id
        labOut[mask] = challenge_id
    Image.fromarray(labOut).save(file_lab_out)
    print(f"Saved converted label: {file_lab_out}")

if __name__ == "__main__":
    file_path = ""
    target_dir = ""
    paths = open(file_path, 'r').read().splitlines()
    repaths = [p.strip().replace('.jpg', '_seg.png') for p in paths]
    for pth in repaths:
        convert_from_ADE(pth, os.path.join(target_dir, pth.split('/')[-1]))
