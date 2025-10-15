
#cat, dog, and 8 other classes 
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os


#grab ten classes from tiny-imagenet

selected_classes = [
    'n02123045',  # cat
    'n02504458',  # elephant
    'n01641577',  # frog
    'n01443537', #fish
    'n01629819', #lizard
    'n01742172', #snake
    'n01855672', #goose
    'n01910747', #jellyfish
    'n01944390', #snail
]

# Define transformations for the images
train_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
])


class ImageNetSubset(Dataset):
    def __init__(self, root_dir, selected_classes, transform=None):
        self.full_dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.selected_classes = selected_classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(selected_classes)}
        
        self.samples = []
        for path, class_idx in self.full_dataset.samples:
            class_name = self.full_dataset.classes[class_idx]
            if class_name in self.selected_classes:
                self.samples.append((path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.full_dataset.loader(path)
        if self.full_dataset.transform is not None:
            sample = self.full_dataset.transform(sample)
        return sample, target


imagenet_root = 'tiny-imagenet-200/train' 

# Create the custom dataset
imagenet_8_classes_dataset = ImageNetSubset(
    root_dir=imagenet_root,
    selected_classes=selected_classes,
    transform=train_transform
)

# Create the DataLoader
batch_size = 32
num_workers = 4 # Adjust based on your system's capabilities
imagenet_8_classes_dataloader = DataLoader(
    imagenet_8_classes_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)


for images, labels in imagenet_8_classes_dataloader:
    print(f"Batch images shape: {images.shape}, Batch labels shape: {labels.shape}")
    break

