import linecache
import os
import cv2
import dlib
import fnmatch
from concurrent.futures import ThreadPoolExecutor
import random
from networks.Transformer import Transformer
import torch
from torchvision import transforms
from configs import parse
import numpy as np
from networks import model_utils
def process_video_OULU( video_path, root_dir, p, t, label, img_use):
    detector = dlib.get_frontal_face_detector()
    video = cv2.VideoCapture(os.path.join(root_dir, img_use, video_path))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frames = random.sample(range(frame_count), 5)
    for frame_number in random_frames:
        video.set(1, frame_number)
        ret, frame = video.read()
        video_name = video_path.split('.')[0]
        img_path = os.path.join(root_dir, p, t.split('.')[0], label, f'{video_name}_{frame_number}.png')
        if ret:
            dets = detector(frame, 1)
            if not dets:
                print(f'Failed to detector face {p}_{t}_{label}_{video_path}')
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0

                if not os.path.exists(os.path.dirname(img_path)):
                    try:
                        os.makedirs(os.path.dirname(img_path))
                    except:
                        print('error')
                face = frame[x1:y1,x2:y2]
                cv2.imwrite(img_path, face)
        else:
            print(f'Failed to read frame {video_name}')
    video.release()

def process_data(root_dir):
    protocol = ['Protocol_1', 'Protocol_2', 'Protocol_3', 'Protocol_4']
    for p in protocol:
        pro_txt = []
        for root, dirs, files in os.walk(os.path.join(root_dir, 'Protocols', p)):
            for file in files:
                if fnmatch.fnmatch(file, '*.txt'):
                    pro_txt.append(file)
        for t in pro_txt:
            with open(os.path.join(root_dir, 'Protocols', p, t), 'r') as file:
                lines = file.readlines()
            video_paths = []
            labels = []
            face_txt_paths = []
            for line in lines:
                parts = line.strip().split(',')
                label = parts[0]
                path = parts[1]
                video_path = path + '.avi'
                video_paths.append(video_path)
                labels.append(label)


            if 'Train' in t:
                img_use = 'Train_files'
            elif 'Test' in t:
                img_use = 'Test_files'
            elif 'Dev' in t:
                img_use = 'Dev_files'
            with ThreadPoolExecutor(max_workers=8) as executor:
                for num in range(len(video_paths)):
                    executor.submit(process_video_OULU(), video_paths[num], root_dir, p, t, labels[num], img_use)


def list_all_files(directory):
    dir = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mov'):
                dir.append(os.path.join(root, file))

    return dir

def cartoon_img(root_dir):
    image_paths = list_all_files(root_dir)
    paths = [path for path in image_paths if "Train" in path]

    args = parse().parse_args()
    model = Transformer()
    model.load_state_dict(torch.load(os.path.join("../pretrained_model", args.style + '_net_G_float.pth')))
    model.eval()
    if args.gpu > -1:
        model.cuda()
    else:
        model.cpu()
    i=0
    for path in paths:
        image=cv2.imread(path)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]

        )
        image=train_transform(image)

        image = image.unsqueeze(0)
        image_g=model(image)
        image_path=path.replace('Train','Train_g')
        image_g=image_g.squeeze()
        image_g = image_g.permute(1, 2, 0)
        image_g = image_g * 255
        print(image_g)
        image_g = image_g.detach().cpu().numpy()

        #image_g = image_g * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image_g = image_g.astype(np.uint8)
        image_g = cv2.cvtColor(image_g, cv2.COLOR_RGB2BGR)
        directory = os.path.dirname(image_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(directory)

        print(image_path)
        cv2.imwrite(image_path,image_g)
        if i%10==0:
            print(f'{i}//{len(image_paths)}')
        i+=1



def remove_img(root_dir):
    path=list_all_files(root_dir)
    for i in range(len(path)):
        try:
            image = cv2.imread(path[i])
            w, h, c = image.shape
            if w < 150 and h < 150:
                os.remove(path[i])
                print(i)
        except:
            print('error')

def process_single_video_Replay(video_path, root_dir, img_use):
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


    random_frames = random.sample(range(frame_count), 5)
    video_split=video_path.split('\\')
    box_name=video_path.split('\\')[-1].split('.')[0]+'.face'
    box_path=os.path.join(root_dir,"face-locations",img_use,video_split[3],video_split[4],box_name)
    folder_path = 'F:/FAS/Replay'
    folder_path = os.path.join(folder_path, img_use,video_split[3],video_split[4])
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
        except:
            print("error")

    for frame_number in random_frames:
        line = linecache.getline(
            box_path,
            frame_number+1)
        location=line.strip().split(' ')
        video.set(1, frame_number)
        ret, frame = video.read()
        video_name = video_path.split("\\")[-1].split('.')[0]
        img_path = os.path.join(folder_path, f'{video_name}_{frame_number}.png')

        if ret:
            bbox_x, bbox_y, bbox_width, bbox_height =int(location[1]),int(location[2]),int(location[3]),int(location[4])
            face = frame[bbox_y:bbox_y+bbox_height, bbox_x:bbox_x+bbox_width]
            cv2.imwrite(img_path, face)
        else:
            print(f'Failed to read frame {video_name}')
    video.release()



root = 'F:\\Replay'
video_path=list_all_files(os.path.join(root,'train'))

process_single_video_Replay(video_path[0],root,'train')
