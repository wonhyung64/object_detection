#%%
import torch
import random
import numpy as np

from torch.nn.functional import pad
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader


def fix_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.nn.Conv2d(3,1,1,).weight
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def collate_fn(batch):
    return tuple(zip(*batch))

#%%
'''configs'''
data_dir = "/Users/wonhyung64/Github/diagnosis/module/mmdetection/data/coco"
data_seed = 0
model_seed = 0
batch_size = 2


'''device'''
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")




'''data load'''
train_set = CocoDetection(root=f"{data_dir}/train2017", annFile=f"{data_dir}/annotations/instances_train2017.json")
test_set = CocoDetection(root=f"{data_dir}/val2017", annFile=f"{data_dir}/annotations/instances_val2017.json")


'''data pipeline'''
fix_seed(data_seed)
train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=collate_fn)


for i, (imgs, annots) in enumerate(train_loader):break
    imgs
    len(annots)

    boxes = []
    labels = []
    object_nums = []
    for annot in annots:
        box = torch.tensor([annot_per_object["bbox"] for annot_per_object in annot])
        boxes.append(box)
        object_nums.append(box.shape[0])
        labels.append(torch.tensor([annot_per_object["category_id"] for annot_per_object in annot]))

    max_object_num = max(object_nums)
    gt_boxes = torch.stack([pad(box, (0, 0, 0, max_object_num - object_nums[i]), mode="constant", value=0.) for i, box in enumerate(boxes)], 0).to(device)
    gt_labels = torch.stack([pad(label, (0, max_object_num - object_nums[i]), mode="constant", value=-1) for i, label in enumerate(labels)], 0).to(device)






torch.allclose()
torch.manual_seed(random_seed)

from torchvision.transforms import transforms

transform = transforms.Compose([
        transforms.Resize((64,256)),
        transforms.ToTensor(),
    ])
transform(img[0]).shape
a = transform(img[1])
transform(img[2]).shape

from torchvision.transforms import ToPILImage
ToPILImage()(a)

# Resize to Fixed img size
# Resize and Pad to Fixed img size
# Pad to max img size

