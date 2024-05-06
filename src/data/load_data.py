
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
            target = Tensor([0, 1])
        else:
            target = Tensor([1,0])

        return (self.data[index][0], target)
    
    
def ingest_data(datapath: str, transform):
    try:
        print("Ingesting data from:", datapath)
        ingested_data = IngestData(datapath, transform)
        print(type(ingested_data.data))
        return ingested_data
    
    except Exception as e:
        raise e