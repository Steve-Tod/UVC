import numpy as np
import time
import random
import scipy.io
from PIL import Image
import cv2
import torch
from os.path import exists, join, split
import libs.transforms_multi as transforms
from torchvision import datasets
import kornia.augmentation as K
import torchvision

def video_loader(video_path, frame_end, step, frame_start=0):
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_start - 1)
    video = []
    for i in range(frame_start - 1, frame_end, step):
        cap.set(1, i)
        success, image = cap.read()
        if not success:
            raise Exception('Error while reading video {}'.format(video_path))
        pil_im = image
        video.append(pil_im)
    return video


def framepair_loader(video_path, frame_start, frame_end):
    
    cap = cv2.VideoCapture(video_path)
    
    pair = []
    id_ = np.zeros(2)
    frame_num = frame_end - frame_start
    if frame_end > 50:
        id_[0] = random.randint(frame_start, frame_end-50)
        id_[1] = id_[0] + random.randint(1, 50)
    else:
        id_[0] = random.randint(frame_start, frame_end)
        id_[1] = random.randint(frame_start, frame_end)

    
    for ii in range(2):
        
        cap.set(1, id_[ii])
        
        success, image = cap.read()
        
        if not success:
            print("id, frame_end:", id_, frame_end)
            raise Exception('Error while reading video {}'.format(video_path))

        h,w,_ = image.shape
        h = (h // 64) * 64
        w = (w // 64) * 64
        image = cv2.resize(image, (w,h))
        image = image.astype(np.uint8)
        pil_im = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        pair.append(pil_im)
            
    return pair

def video_frame_counter(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap.get(7)

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)


class VidListv1(torch.utils.data.Dataset):
    # for warm up, random crop both
    def __init__(self, video_path, list_path, patch_size, rotate = 10, scale=1.2, is_train=True, moreaug= True):
        super(VidListv1, self).__init__()
        self.data_dir = video_path
        self.list_path = list_path
        normalize = transforms.Normalize(mean = (128, 128, 128), std = (128, 128, 128))

        t = []
        if rotate > 0:
            t.append(transforms.RandomRotate(rotate))
        if scale > 0:
            t.append(transforms.RandomScale(scale))
        t.extend([transforms.RandomCrop(patch_size, seperate =moreaug), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
              normalize])

        self.transforms = transforms.Compose(t)
        
        self.is_train = is_train
        self.read_list()

    def __getitem__(self, idx):
        while True:
            video_ = self.list[idx]
            frame_end = video_frame_counter(video_)-1
            if frame_end <=0:
                print("Empty video {}, skip to the next".format(self.list[idx]))
                idx += 1
            else:
                break

        pair_ = framepair_loader(video_, 0, frame_end)
        data = list(self.transforms(*pair_))
        return tuple(data)

    def __len__(self):
        return len(self.list)

    def read_list(self):
        path = join(self.list_path)
        root = path.partition("Kinetices/")[0]
        if not exists(path):
            raise Exception("{} does not exist in kinet_dataset.py.".format(path))
        self.list = [line.replace("/Data/", root).strip() for line in open(path, 'r')]


class VidListv2(torch.utils.data.Dataset):
    # for localization, random crop frame1
    def __init__(self, video_path, list_path, patch_size, window_len, rotate = 10, scale = 1.2, full_size = 640, is_train=True):
        super(VidListv2, self).__init__()
        self.data_dir = video_path
        self.list_path = list_path
        self.window_len = window_len
        normalize = transforms.Normalize(mean = (128, 128, 128), std = (128, 128, 128))
        self.transforms1 = transforms.Compose([
                           transforms.RandomRotate(rotate),
                           # transforms.RandomScale(scale),
                           transforms.ResizeandPad(full_size),
                           transforms.RandomCrop(patch_size),
                           transforms.ToTensor(),
                           normalize])          
        self.transforms2 = transforms.Compose([
                           transforms.ResizeandPad(full_size),
                           transforms.ToTensor(),
                           normalize])
        self.is_train = is_train
        self.read_list()

    def __getitem__(self, idx):
        while True:
            video_ = self.list[idx]
            frame_end = video_frame_counter(video_)-1
            if frame_end <=0:
                print("Empty video {}, skip to the next".format(self.list[idx]))
                idx += 1
            else:
                break

        pair_ = framepair_loader(video_, 0, frame_end)
        data1 = list(self.transforms1(*pair_))
        data2 = list(self.transforms2(*pair_))
        if self.window_len == 2:
            data = [data1[0],data2[1]]
        else:
            data = [data1[0],data2[1], data2[2]]
        return tuple(data)

    def __len__(self):
        return len(self.list)

    def read_list(self):
        path = join(self.list_path)
        root = path.partition("Kinetices/")[0]
        if not exists(path):
            raise Exception("{} does not exist in kinet_dataset.py.".format(path))
        self.list = [line.replace("/Data/", root).strip() for line in open(path, 'r')]
        
        
class VidListv3(torch.utils.data.Dataset):
    # for CRW, return a sequence
    def __init__(self,
                 video_path,
                 list_path,
                 size,
                 patch_size,
                 seq_len,
                 frame_rate,
                 is_train=True):
        super().__init__()
        self.data_dir = video_path
        self.list_path = list_path
        t = []
        normalize = torchvision.transforms.Normalize(mean=(0.45, 0.45, 0.45),
                                                     std=(0.225, 0.225, 0.225))
        t.append(torchvision.transforms.Resize((size, size)))
        t.append(torchvision.transforms.ToTensor())
        t.append(normalize)

        self.transforms = torchvision.transforms.Compose(t)
        self.to_image = torchvision.transforms.ToPILImage()
        # create patches
        self.unfold = torch.nn.Unfold(
            (patch_size, patch_size),
            stride=(patch_size // 2, patch_size // 2))
        self.spatial_jitter = K.RandomResizedCrop(size=(patch_size,
                                                        patch_size),
                                                  scale=(0.7, 0.9),
                                                  ratio=(0.7, 1.3))

        self.is_train = is_train
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.frame_rate = frame_rate

        self.read_list()

    def __getitem__(self, idx):
        while True:
            video_path = self.list[idx]
            frame_count, fps = get_video_info(video_path)
            rel_rate = int(fps / self.frame_rate)  # relative rate
            if frame_count <= self.seq_len * rel_rate:
                print("Too short video {}, length {}, skip to the next".format(
                    self.list[idx], int(frame_count)))
                idx += 1
            else:
                break

        frame_start = torch.randint(low=0,
                                    high=int(frame_count -
                                             self.seq_len * rel_rate),
                                    size=(1, )).item()
        frames = video_loader(video_path,
                              frame_start + self.seq_len * rel_rate, rel_rate,
                              frame_start)
        frames = frames[:self.seq_len]
        frame_tensor_list = []
        for f in frames:
            frame_img = Image.fromarray(f[:, :, ::-1])
            frame_tensor = self.transforms(Image.fromarray(f[:, :, ::-1]))
            frame_tensor_list.append(frame_tensor)
        frames_tensor = torch.stack(frame_tensor_list, dim=0)
        with torch.no_grad():
            patches_tensor = self.unfold(frames_tensor)
            patches_tensor = patches_tensor.view(self.seq_len, 3, self.patch_size, self.patch_size, -1)
            patches = []
            for i in range(patches_tensor.size(-1)):
                patches.append(self.spatial_jitter(patches_tensor[:, :, :, :, i]))
            patches_tensor = torch.stack(patches, 2)
        # C*T*H*W, C*T*P*h*w
        return frames_tensor.transpose(0, 1).contiguous(), patches_tensor.transpose(0, 1).contiguous()

    def __len__(self):
        return len(self.list)

    def read_list(self):
        path = join(self.list_path)
        root = path.partition("Kinetices/")[0]
        if not exists(path):
            raise Exception(
                "{} does not exist in kinet_dataset.py.".format(path))
        self.list = [
            line.replace("/Data/", root).strip() for line in open(path, 'r')
        ]

    def detrans(self, tensor):
        img_list = []
        for i in range(tensor.size(0)):
            img_tensor = tensor[i]  # 3, 256, 256
            img_tensor *= torch.ones(3, 1, 1) * 0.225
            img_tensor += 0.45
            img_list.append(self.to_image(img_tensor))
        return img_list