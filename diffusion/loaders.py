import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import configs


size = configs.IMG_SIZE
batch_size = configs.BATCH_SIZE

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size),
    transforms.Lambda(lambda t: (t * 2) - 1),
    transforms.RandomHorizontalFlip(),
])
reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
])

anime_loader = DataLoader(ImageFolder(configs.ANIME_PATH, transform=transform), batch_size=batch_size, shuffle=True, num_workers=0)
human_loader = DataLoader(ImageFolder(configs.HUMAN_PATH, transform=transform), batch_size=batch_size, shuffle=True, num_workers=0)