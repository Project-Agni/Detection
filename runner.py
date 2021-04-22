import argparse

import gym
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torchvision import datasets, transforms

import gym_env
from agents import cnn, dqn

gym_env.dummy()  # Calls __init__


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--arch", type=str, default="dqn", help="Can be either of CNN, DQN"
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size, "drop_last": True}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.4363, 0.3613, 0.3098), (0.2360, 0.2087, 0.1925)),
        ]
    )

    dataset = datasets.ImageFolder("datasets/train", transform=transform)
    num_images = len(dataset)
    train_split = int(0.8 * num_images)
    val_split = num_images - train_split
    dataset1, dataset2 = data.random_split(dataset, (train_split, val_split))

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if args.arch.lower() == "dqn":
        # Initialization of the environment
        env = gym.make(
            "ProjectAgni-v0", train_loader=train_loader, test_loader=test_loader
        )
        _ = env.reset()

        # online_model = dqn.QNet().to(device)
        # target_model = dqn.QNet().to(device)
        model = dqn.QNet().to(device)
    elif args.arch.lower() == "cnn":
        env = None
        model = cnn.CNN().to(device)
    else:
        raise NotImplementedError

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        model.train_model(args, model, device, train_loader, optimizer, epoch, env=env)
        model.test_model(model, device, test_loader, env)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), f"project_agni_{args.arch}.pt")


if __name__ == "__main__":
    main()
