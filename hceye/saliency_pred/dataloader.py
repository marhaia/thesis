from torchvision import transforms, utils
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import torch
import os, cv2
import random

class HighlightDataset_(DataLoader):
    def __init__(self, img_dir, gt_dir, fix_dir, img_list, exten='.png'):
        print("Debug: Initializing HighlightDataset")
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.fix_dir = fix_dir
        self.img_list = img_list  # List of image names
        self.exten = exten
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, idx):
        img_id = self.img_list[idx]

        img_path = os.path.join(self.img_dir, img_id + self.exten) if not img_id.endswith(self.exten) else os.path.join(self.img_dir, img_id)
        gt_path = os.path.join(self.gt_dir, img_id + self.exten) if not img_id.endswith(self.exten) else os.path.join(self.gt_dir, img_id)
        fix_path = os.path.join(self.fix_dir, img_id + self.exten) if not img_id.endswith(self.exten) else os.path.join(self.fix_dir, img_id)

        img_1 = Image.open(img_path).convert('RGB')
        gt = np.array(Image.open(gt_path).convert('L'), dtype=np.float32)
        fixations = np.array(Image.open(fix_path).convert('L'), dtype=np.float32)

        # Resize and normalize ground truth and fixations
        gt = cv2.resize(gt, (256, 256)) / 255.0
        fixations = cv2.resize(fixations, (256, 256)) > 0.5

        # Apply transformations to the image
        img_1 = self.img_transform(img_1)

        # Create a zero tensor for img_2 with the same dimensions as img_1
        img_2 = torch.zeros_like(img_1)

        # Return img_1 tensor, the zero tensor for img_2, along with ground truth and fixations
        return img_1, img_2, torch.from_numpy(gt), torch.from_numpy(fixations).float()

    def __len__(self):
        return len(self.img_list)


class HighlightDataset(DataLoader):
    def __init__(self, img_dir, gt_dir, fix_dir, img_pairs, exten='.png'):
        print("Debug: Initializing SaliconDataset")
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.fix_dir = fix_dir
        self.img_pairs = img_pairs  # List of tuples where each tuple has the names of the pair images
        self.exten = exten
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, idx):
        img_id_1, img_id_2 = self.img_pairs[idx]

        # Assuming img_id is without the '_dynamic_1' or '_dynamic_2' suffix
        img_base_id = img_id_1.rsplit('_', 1)[0]

        # Here, check if the img_id_1 already has '.png' and concatenate accordingly
        if not img_id_1.endswith(self.exten):
            img_path_1 = os.path.join(self.img_dir, img_id_1 + self.exten)
        else:
            img_path_1 = os.path.join(self.img_dir, img_id_1)

        # Do the same check for img_id_2
        if not img_id_2.endswith(self.exten):
            img_path_2 = os.path.join(self.img_dir, img_id_2 + self.exten)
        else:
            img_path_2 = os.path.join(self.img_dir, img_id_2)

        # Check and concatenate for gt_path
        if not img_base_id.endswith(self.exten):
            gt_path = os.path.join(self.gt_dir, img_base_id + self.exten)
        else:
            gt_path = os.path.join(self.gt_dir, img_base_id)

        # Check and concatenate for fix_path
        if not img_base_id.endswith(self.exten):
            fix_path = os.path.join(self.fix_dir, img_base_id + self.exten)
        else:
            fix_path = os.path.join(self.fix_dir, img_base_id)

        img_1 = Image.open(img_path_1).convert('RGB')
        img_2 = Image.open(img_path_2).convert('RGB')

        gt = np.array(Image.open(gt_path).convert('L'), dtype='float')
        fixations = np.array(Image.open(fix_path).convert('L'), dtype='float')

        # Resize and normalize ground truth and fixations
        gt = cv2.resize(gt, (256, 256)) / 255.0
        fixations = cv2.resize(fixations, (256, 256)) > 0.5

        # Apply transformations to each image separately
        img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img_1 = img_transform(img_1)
        img_2 = img_transform(img_2)

        # Return separate tensors for each image, along with ground truth and fixations
        return img_1, img_2, torch.FloatTensor(gt), torch.FloatTensor(fixations)
    def __len__(self):
        return len(self.img_pairs)


class NewDataset(DataLoader):

    def __init__(self, img_dir, gt_dir, fix_dir, img_ids, split='train', exten='.png'):
        self.img_dir = os.path.join(os.path.join(img_dir, split))
        self.gt_dir = gt_dir
        self.fix_dir = fix_dir
        self.img_ids = img_ids
        self.split = split
        self.exten = exten
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + self.exten)

        gt_subfolders = ['labels_abs', 'labels_low', 'labels_high', 'labels_all']
        fix_subfolders = ['fixs_abs', 'fixs_low', 'fixs_high', 'fixs_all']

        gts = []
        fixs = []

        for gt_sub, fix_sub in zip(gt_subfolders, fix_subfolders):
            gt_path = os.path.join(self.gt_dir, gt_sub, self.split, img_id + self.exten)
            fix_path = os.path.join(self.fix_dir, fix_sub, self.split, img_id + self.exten)

            gt = np.array(Image.open(gt_path).convert('L'))
            gt = gt.astype('float')
            gt = cv2.resize(gt, (256, 256))

            fixations = np.array(Image.open(fix_path).convert('L'))
            fixations = fixations.astype('float')

            if np.max(gt) > 1.0:
                gt = gt / 255.0
            fixations = (fixations > 0.5).astype('float')

            gts.append(torch.FloatTensor(gt))
            fixs.append(torch.FloatTensor(fixations))

        img = Image.open(img_path).convert('RGB')
        img = self.img_transform(img)

        return img, gts, fixs

    def __len__(self):
        return len(self.img_ids)

class SaliconDataset(DataLoader):
    def __init__(self, img_dir, gt_dir, fix_dir, img_ids, exten='.png', highlight_mode='none'):
        print("Debug: Initializing SaliconDataset")
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.fix_dir = fix_dir
        # Filter img_ids based on highlight_mode
        if highlight_mode == 'none':
            self.img_ids = [img_id for img_id in img_ids if '_highlight' not in img_id]
        elif highlight_mode == 'highlight':
            self.img_ids = [img_id for img_id in img_ids if '_highlight' in img_id]
        else:  # 'both' or any other value will not filter out any ids
            self.img_ids = img_ids
        self.exten = exten
        self.highlight_mode = highlight_mode
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

    def get_img_ids(self):
        # Method to retrieve the current list of img_ids
        return self.img_ids


    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img_path = os.path.join(self.img_dir, img_id + self.exten)
        gt_path = os.path.join(self.gt_dir, img_id + self.exten)
        fix_path = os.path.join(self.fix_dir, img_id + self.exten)

        img = Image.open(img_path).convert('RGB')

        gt = np.array(Image.open(gt_path).convert('L'))
        gt = gt.astype('float')
        gt = cv2.resize(gt, (256, 256))

        fixations = np.array(Image.open(fix_path).convert('L'))
        fixations = fixations.astype('float')
        fixations = cv2.resize(fixations, (256, 256))

        img = self.img_transform(img)
        if np.max(gt) > 1.0:
            gt = gt / 255.0
        fixations = (fixations > 0.5).astype('float')

        assert np.min(gt) >= 0.0 and np.max(gt) <= 1.0
        return img, torch.FloatTensor(gt), torch.FloatTensor(fixations)

    def __len__(self):
        return len(self.img_ids)

class DynamicDataset(DataLoader):
    def __init__(self, img_dir, gt_dir, fix_dir, mode='highlighted', transform=None):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.fix_dir = fix_dir
        self.mode = mode
        self.img_ids = [os.path.splitext(file)[0] for file in os.listdir(img_dir) if 'dynamic_1' in file]
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, idx):
        base_id = self.img_ids[idx].rsplit('_', 1)[0]

        if self.mode == 'highlighted':
            img_id = base_id + '_2'  # Assuming the highlighted image ends with '_2'
        elif self.mode == 'random':
            img_id = base_id + ('_1' if random.random() < 0.5 else '_2')  # Randomly choose between '_1' or '_2'
        elif self.mode == 'pair':
            # In 'pair' mode, we'll return a tuple of both images
            img_id_1 = base_id + '_1'
            img_id_2 = base_id + '_2'
            img_path_1 = os.path.join(self.img_dir, img_id_1 + '.png')
            img_path_2 = os.path.join(self.img_dir, img_id_2 + '.png')
            img_1 = Image.open(img_path_1).convert('RGB')
            img_2 = Image.open(img_path_2).convert('RGB')
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
            # Load and process the ground truth and fixations once as they are the same for both images
            gt_path = os.path.join(self.gt_dir, base_id + '.png')
            fix_path = os.path.join(self.fix_dir, base_id + '.png')
            gt = np.array(Image.open(gt_path).convert('L'), dtype=np.float32)
            fixations = np.array(Image.open(fix_path).convert('L'), dtype=np.float32)
            gt = cv2.resize(gt, (256, 256)) / 255.0
            fixations = cv2.resize(fixations, (256, 256))
            fixations = (fixations > 0.5).astype('float32')
            return (img_1, img_2), torch.FloatTensor(gt), torch.FloatTensor(fixations)
        else:
            raise ValueError("Invalid mode. Options are: 'highlighted', 'random', 'pair'")

        img_path = os.path.join(self.img_dir, img_id + '.png')
        gt_path = os.path.join(self.gt_dir, base_id + '.png')  # Removing the extra '_dynamic' in the filename
        fix_path = os.path.join(self.fix_dir, base_id + '.png')

        img = Image.open(img_path).convert('RGB')
        gt = np.array(Image.open(gt_path).convert('L'), dtype=np.float32)
        fixations = np.array(Image.open(fix_path).convert('L'), dtype=np.float32)

        img = self.transform(img)
        gt = cv2.resize(gt, (256, 256)) / 255.0
        fixations = cv2.resize(fixations, (256, 256))
        fixations = (fixations > 0.5).astype('float32')

        return img, torch.FloatTensor(gt), torch.FloatTensor(fixations)

    def __len__(self):
        # In 'pair' mode, the length is effectively half since we're dealing with pairs
        return len(self.img_ids) if self.mode != 'pair' else len(self.img_ids) // 2

# # Ensure the base directory ends with a slash
# base_dir = r"C:\Users\wuzek\OneDrive\Desktop\NewProject\highlight_data\dynamic"
#
# train_img_dir = os.path.join(base_dir, "images/train/")
# train_gt_dir = os.path.join(base_dir, "maps/train/")
# train_fix_dir = os.path.join(base_dir, "fixations/train/")
#
# val_img_dir = os.path.join(base_dir, "images/val/")
# val_gt_dir = os.path.join(base_dir, "maps/val/")
# val_fix_dir = os.path.join(base_dir, "fixations/val/")
#
# # Retrieve all image files from the directory, sorted to ensure pairing
# all_image_files = sorted([f for f in os.listdir(train_img_dir) if not f.startswith('.')])
# all_image_files_ = sorted([f for f in os.listdir(val_img_dir) if not f.startswith('.')])
#
# train_img_ids = [nm.split(".")[0] for nm in os.listdir(train_img_dir)]
# val_img_ids = [nm.split(".")[0] for nm in os.listdir(val_img_dir)]
#
#
# # Create the dataset for highlighted images
# highlighted_dataset = DynamicDataset(img_dir=train_img_dir , gt_dir=train_gt_dir, fix_dir=train_fix_dir, mode='highlighted')
#
# # For random selection
# random_dataset = DynamicDataset(img_dir=train_img_dir, gt_dir=train_gt_dir , fix_dir=train_fix_dir , mode='random')
#
# # For pairs
# pair_dataset = DynamicDataset(img_dir=train_img_dir , gt_dir=train_gt_dir , fix_dir=train_fix_dir , mode='pair')
#
#
# def test_dataset(dataset, num_samples_to_test=5):
#     for i in range(num_samples_to_test):
#         sample = dataset[i]
#
#         if dataset.mode == 'pair':
#             img_pair, gt, fixations = sample
#             print(f"Sample {i}: Image 1 Shape: {img_pair[0].shape}, Image 2 Shape: {img_pair[1].shape}")
#             print(f"GT Shape: {gt.shape}, Fixations Shape: {fixations.shape}")
#         else:
#             img, gt, fixations = sample
#             print(f"Sample {i}: Image Shape: {img.shape}")
#             print(f"GT Shape: {gt.shape}, Fixations Shape: {fixations.shape}")
#
#
# # Test the datasets
# print("Testing highlighted dataset:")
# test_dataset(highlighted_dataset)
#
# print("\nTesting random dataset:")
# test_dataset(random_dataset)
#
# print("\nTesting pair dataset:")
# test_dataset(pair_dataset)