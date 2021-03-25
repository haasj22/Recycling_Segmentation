import os
import sys
from torch.utils.data import ConcatDataset, Sampler

from IIC.code.datasets.clustering.truncated_dataset import TruncatedDataset
from IIC.code.utils.cluster.transforms import sobel_make_transforms, \
    greyscale_make_transforms
from IIC.code.utils.semisup.dataset import TenCropAndFinish

def create_basic_clustering_dataloaders(config):
  """
  My original data loading code is complex to cover all my experiments. Here is a simple version.
  Use it to replace cluster_twohead_create_dataloaders() in the scripts.
  
  This uses ImageFolder but you could use your own subclass of torch.utils.data.Dataset.
  (ImageFolder data is not shuffled so an ideally deterministic random sampler is needed.)
  
  :param config: Requires num_dataloaders and values used by *make_transforms(), e.g. crop size, 
  input size etc.
  :return: Training and testing dataloaders
  """

  # Change these according to your data:
  greyscale = False
  train_data_path = os.path.join(config.dataset_root)
  test_val_data_path = os.path.join(config.dataset_root)
  test_data_path = os.path.join(config.dataset_root)
  assert (config.batchnorm_track)  # recommended (for test time invariance to batch size)

  # Transforms:
  if greyscale:
    tf1, tf2, tf3 = greyscale_make_transforms(config)
  else:
    tf1, tf2, tf3 = sobel_make_transforms(config)

  # Training data:
  # main output head (B), auxiliary overclustering head (A), same data for both
  dataset_head_B = torchvision.datasets.ImageFolder(root=train_data_path, transform=tf1),
  datasets_tf_head_B = [torchvision.datasets.ImageFolder(root=train_data_path, transform=tf2)
                        for _ in range(config.num_dataloaders)]
  dataloaders_head_B = [torch.utils.data.DataLoader(
    dataset_head_B,
    batch_size=config.dataloader_batch_sz,
    shuffle=False,
    sampler=DeterministicRandomSampler(dataset_head_B),
    num_workers=0,
    drop_last=False)] + \
                       [torch.utils.data.DataLoader(
                         datasets_tf_head_B[i],
                         batch_size=config.dataloader_batch_sz,
                         shuffle=False,
                         sampler=DeterministicRandomSampler(datasets_tf_head_B[i]),
                         num_workers=0,
                         drop_last=False) for i in range(config.num_dataloaders)]

  dataset_head_A = torchvision.datasets.ImageFolder(root=train_data_path, transform=tf1)
  datasets_tf_head_A = [torchvision.datasets.ImageFolder(root=train_data_path, transform=tf2)
                        for _ in range(config.num_dataloaders)]
  dataloaders_head_A = [torch.utils.data.DataLoader(
    dataset_head_A,
    batch_size=config.dataloader_batch_sz,
    shuffle=False,
    sampler=DeterministicRandomSampler(dataset_head_A),
    num_workers=0,
    drop_last=False)] + \
                       [torch.utils.data.DataLoader(
                         datasets_tf_head_A[i],
                         batch_size=config.dataloader_batch_sz,
                         shuffle=False,
                         sampler=DeterministicRandomSampler(datasets_tf_head_A[i]),
                         num_workers=0,
                         drop_last=False) for i in range(config.num_dataloaders)]

  # Testing data (labelled):
  mapping_assignment_dataloader, mapping_test_dataloader = None, None
  if os.path.exists(test_data_path):
    mapping_assignment_dataset = torchvision.datasets.ImageFolder(test_val_data_path, transform=tf3)
    mapping_assignment_dataloader = torch.utils.data.DataLoader(
      mapping_assignment_dataset,
      batch_size=config.batch_sz,
      shuffle=False,
      sampler=DeterministicRandomSampler(mapping_assignment_dataset),
      num_workers=0,
      drop_last=False)

    mapping_test_dataset = torchvision.datasets.ImageFolder(test_data_path, transform=tf3)
    mapping_test_dataloader = torch.utils.data.DataLoader(
      mapping_test_dataset,
      batch_size=config.batch_sz,
      shuffle=False,
      sampler=DeterministicRandomSampler(mapping_test_dataset),
      num_workers=0,
      drop_last=False)

  return dataloaders_head_A, dataloaders_head_B, \
         mapping_assignment_dataloader, mapping_test_dataloader

class DeterministicRandomSampler(Sampler):
  # Samples elements randomly, without replacement - same order every time.

  def __init__(self, data_source):
    self.data_source = data_source
    self.gen = torch.Generator().manual_seed(0)

  def __iter__(self):
    return iter(torch.randperm(len(self.data_source), generator=self.gen).tolist())

  def __len__(self):
    return len(self.data_source)
