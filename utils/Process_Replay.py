import linecache
import os
import random
import threading
import cv2

def list_all_files(directory):
    dir = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                dir.append(os.path.join(root, file))

    return dir

def process_single_video_Replay(video_path, root_dir, img_use):
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


    random_frames = random.sample(range(frame_count), 5)
    if video_path.find('real'):
        video_split=video_path.split('\\')
        box_name=video_path.split('\\')[-1].split('.')[0]+'.face'
        box_path=os.path.join(root_dir,"face-locations",img_use,video_split[3],box_name)
        folder_path = 'F:/FAS/Replay'
        folder_path = os.path.join(folder_path, img_use,video_split[3])

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
            try:
                bbox_x, bbox_y, bbox_width, bbox_height =int(location[1]),int(location[2]),int(location[3]),int(location[4])
                face = frame[bbox_y:bbox_y+bbox_height, bbox_x:bbox_x+bbox_width]
                cv2.imwrite(img_path, face)
            except:
                print("save error")
        else:
            print(f'Failed to read frame {video_name}')
    video.release()

def process_replay():
    root = 'F:\\Replay'
    train_video_path = list_all_files(os.path.join(root, 'train','real'))
    print(train_video_path)
    test_video_path=list_all_files(os.path.join(root,'test','real'))
    print(train_video_path)
    for path in train_video_path:
        process_single_video_Replay(path, root, 'train')

    for path in test_video_path:
        process_single_video_Replay(path, root, 'test')

def thread_function(video_paths, root_dir, img_use):
    for path in video_paths:
        process_single_video_Replay(path, root_dir, img_use)

def process_replay_multithreaded():
    root = 'F:\\Replay'
    train_video_paths = list_all_files(os.path.join(root, 'train'))
    test_video_paths = list_all_files(os.path.join(root, 'test'))

    # 为训练和测试视频路径创建线程
    train_thread = threading.Thread(target=thread_function, args=(train_video_paths, root, 'train'))
    test_thread = threading.Thread(target=thread_function, args=(test_video_paths, root, 'test'))

    # 启动线程
    train_thread.start()
    test_thread.start()

    # 等待线程完成
    train_thread.join()
    test_thread.join()

# process_replay_multithreaded()


test=list_all_files('F:\\FAS\\Replay\\train')
print(test)