import numpy as np
import cv2
import os
import json

def make_dataset(split_file, split, root, num_classes, depth_size):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    for vid in data.keys():
        if data[vid]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(root, vid)):
            continue

        num_frames = len(os.listdir(os.path.join(root, vid)))

        if num_frames < depth_size:
            continue

        label = np.zeros((num_classes, num_frames), np.float32)
        fps = float(num_frames)/data[vid]['duration']

        for ann in data[vid]['actions']:
            for fr in range(0, num_frames, 1):
                if fr/fps > ann[1] and fr/fps < ann[2]:
                    label[ann[0], fr] = 1 # binary classification
        dataset.append((vid, label, data[vid]['duration'], num_frames))

    return dataset

def get_key_vid(split_file):
    vid_keys = {}
    with open(split_file, 'r') as f:
        data = json.load(f)

    for vid in data.keys():

        vid_key = vid.strip().split("/")[0]
        if vid_key not in vid_keys:
            vid_keys[vid_key] = data[vid]['actions'][0][0]

    index_vid = {}
    for key in vid_keys:
        index_vid[vid_keys[key]] = key
    return index_vid

def load_rgb_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num):
    img = cv2.imread(os.path.join(image_dir, vid, str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
    w,h,c = img.shape
    if w < 256 or h < 256:
        sc = 256./min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
    img = (img/255.)*2 - 1
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)


