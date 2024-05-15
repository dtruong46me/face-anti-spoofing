import argparse
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from load_data import IngestData  # Giả sử bạn đã lưu lớp IngestData trong file ingest_data.py

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
    plt.savefig()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Data')
    parser.add_argument('--datapath', type=str, required=True, help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = IngestData(args.datapath, transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    visualize_data(data_loader)
