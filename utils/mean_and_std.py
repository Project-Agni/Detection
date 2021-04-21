from torch.utils import data
from torchvision import datasets, transforms


def main():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder("../datasets/train", transform=transform)
    loader = data.DataLoader(dataset, batch_size=10, num_workers=0, shuffle=False)

    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_samples = images.size(
            0
        )  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(dataset)
    std /= len(dataset)

    print(f"Mean: {mean} \t Std: {std}")


if __name__ == "__main__":
    """
    Output: 
    Mean: tensor([0.4363, 0.3613, 0.3098]) 	 Std: tensor([0.2360, 0.2087, 0.1925])
    """
    main()
