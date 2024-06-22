import argparse
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from load_data import IngestData  # Giả sử bạn đã lưu lớp IngestData trong file ingest_data.py

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#sys.path.insert(0, path)

from utils import load_transform_2

def visualize_data(data_loader, num_images=5):
    data_iter = iter(data_loader)
    fig, axes = plt.subplots(1, num_images, figsize=(20, 5))

    for i in range(num_images):
        images, labels = next(data_iter)
        image = images[0].permute(1, 2, 0).numpy()  # Chuyển đổi tensor thành numpy array và điều chỉnh trật tự các kênh màu
        label = labels[0].numpy()

        ax = axes[i]
        ax.imshow(image)
        ax.set_title(f"Label: {label}\nTensor: {labels[0].tolist()}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig("images.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Data')
    parser.add_argument('--datapath', type=str, required=True, help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to visualize')
    args = parser.parse_args()

    transform = load_transform_2()

    dataset = IngestData(args.datapath, transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    visualize_data(data_loader, num_images=args.num_images)
