from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def create_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int
):
    training_data = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True)

    return train_dataloader
