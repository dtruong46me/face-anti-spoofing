
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch import tensor

class IngestData(Dataset):
    def __init__(self, datapath: str, transform) -> None:
        self.datapath = datapath
        self.transform = transform
        self.data = ImageFolder(self.datapath, transform=self.transform)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """
        0: real -> [1,0] #R
        1: fake -> [0,1] #R
        """
        image, label = self.data[index]
        target = tensor([0,1]) if label != 0 else tensor([1,0])
        return (image, target)
    
    
def ingest_data(datapath: str, transform):
    try:
        print(" > Ingesting data from:", datapath)

        ingested_data = IngestData(datapath, transform)
        return ingested_data
    
    except Exception as e:
        raise e