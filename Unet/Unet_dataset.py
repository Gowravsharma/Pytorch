import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class CarvanaDataset(Dataset):
  def __init__(self, root_path, test = False):
    self.test = test
    self.root_path = root_path
    if self.test:
      self.images = sorted([fr'{root_path}\test\{i}' for i in os.listdir(fr'{root_path}\test')])
      mask = None
    else:
      self.images = sorted([fr'{root_path}\train\{i}' for i in os.listdir(fr'{root_path}\train')])
      self.masks = sorted([fr'{root_path}\train_masks\{i}' for i in os.listdir(fr'{root_path}\train_masks')])

    self.transform = transforms.Compose([
      transforms.Resize((512,512)),
      transforms.ToTensor()
    ])

  def __getitem__(self,index):
    img = Image.open(self.images[index]).convert("RGB")
    img = self.transform(img)
    
    if self.test:
      return img
    
    else:
      mask = Image.open(self.masks[index]).convert('L')
      mask = self.transform(mask)
      return img, mask
  
  def __len__(self):
    return len(self.images)

if __name__ == '__main__':
  path = fr'C:\V\Sem_4_notes\Pytorch\carvana_maskig_data'
  example = CarvanaDataset(path)
  print(len(example))