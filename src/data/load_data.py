
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch import Tensor

class IngestData(Dataset):
    def __init__(self, datapath: str, transform) -> None:
        self.datapath = datapath
        self.transform = transform
        self.data = ImageFolder(self.datapath, transform=self.transform)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.data[index][1]==0:
            target = [0, 1]
        else:
            target = [1,0]

        return (self.data[index], target)
    
    
def ingest_data(datapath: str, transform):
    try:
        return IngestData(datapath, transform)
    
    except Exception as e:
        raise e