import os
import PIL

import functools

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import models

from trainers import Trainer

import data



def video_transform(video, image_transform):
    vid = []
    for im in video:
        vid.append(image_transform(im))

    vid = torch.stack(vid).permute(1,0,2,3)

    return vid

if __name__ == "__main__":

    n_channels = 3

    image_transforms = transforms.Compose([
        PIL.Image.fromarray,
        transforms.Scale(64),
        transforms.ToTensor(),
        lambda x: x[:n_channels, ::],
        transforms.Normalize((0.5,0.5, 0.5), (0.5,0.5,0.5)),
    ])

    video_transforms = functools.partial(video_transform, image_transform = image_transforms)

    video_length = 16
    image_batch = 32
    video_batch = 32

    dim_z_content = 50
    dim_z_motion = 10
    dim_z_category = 4

    data_path = '../data/actions'
    log_path = '../logs'
    dataset = data.VideoFolderDataset(data_path)
    image_dataset = data.ImageDataset(dataset, image_transforms)
    image_loader = DataLoader(image_dataset, batch_size=image_batch, drop_last=True, num_workers=2, shuffle=True)

    video_dataset = data.VideoDataset(dataset, 16, 2, video_transforms)
    video_loader = DataLoader(video_dataset, batch_size=video_batch, drop_last=True, num_workers=2, shuffle=True)

    generator = models.VideoGenerator(n_channels, dim_z_content, dim_z_category, dim_z_motion, video_length)

    image_discriminator = models.PatchImageDiscriminator(n_channels=n_channels, use_noise=True, noise_sigma=0.1)

    video_discriminator = models.CategoricalVideoDiscriminator(dim_categorical=dim_z_category, n_channels=n_channels, use_noise=True, noise_sigma=0.2)

    if torch.cuda.is_available():
        generator.cuda()
        image_discriminator.cuda()
        video_discriminator.cuda()

    trainer = Trainer(image_loader, video_loader,
                        log_interval=100,
                        train_batches=100000,
                        log_folder=log_path,
                        use_cuda=torch.cuda.is_available(),
                        use_infogan=True,
                        use_categories=True)

    trainer.train(generator, image_discriminator, video_discriminator)
