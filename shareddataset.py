from multiprocessing import shared_memory
import torch, itertools
from torch.utils.data import Dataset

__all__ = ['SharedDataset']

class SharedDataset(Dataset):
  """Dataset with lock-free shared memory cache (reused across processes).

  Wraps another Dataset, such that:
  - It gets samples from the wrapped Dataset the first time they're requested.
  - Afterwards, they are obtained from a shared memory cache.
  The memory cache uses multiprocessing.shared_memory.SharedMemory, so multiple
  processes will reuse it -- this is especially useful for very large datasets.
  
  To prevent unnecessary copies, the returned tensors point to the same shared
  memory. If you need to modify them in-place, copy them first, otherwise the
  buffer will be corrupted.

  Args:
    dataset (Dataset): A PyTorch Dataset to be wrapped. It needs to return either
      a Tensor or a tuple/list of Tensors.
    shared_mem_name (str): A name that identifies this shared memory buffer. Must
      be unique for each dataset, otherwise they will reuse the same memory and
      the buffer will be corrupted.

  Author: Joao F. Henriques
  """
  def __init__(self, dataset, shared_mem_name):
    self.dataset = dataset
    self.shared_mem_name = shared_mem_name
    self._initialize()
  
  def _initialize(self):
    """Internal method to initialize the shared memory."""
    # get the first sample to figure out the returned layout and memory use
    sample = self.dataset[0]
    if torch.is_tensor(sample):  # wrap in a tuple
      sample = (sample,)
    try:  # ensure it's a tuple with tensors
      if not all(torch.is_tensor(e) for e in sample):
        raise TypeError()
    except TypeError:
      raise TypeError("SharedDataset: The wrapped dataset must return a Tensor or an iterable with Tensors.")
    
    self.sample_shapes = [e.shape for e in sample]
    self.sample_dtypes = [e.dtype for e in sample]
    sample_sizes = [len(e.numpy().tobytes()) for e in sample]

    # store offset where each tensor starts, in bytes.
    # e.g.: for 2 tensors of size 3, will contain [1, 4, 7] (the first byte is reserved)
    self.sample_offsets = list(itertools.accumulate([1] + sample_sizes))

    # total size of one sample
    total_sample_size = self.sample_offsets[-1]

    try:
      # load existing shared memory buffer if possible
      self.mem = shared_memory.SharedMemory(name=self.shared_mem_name, create=False)

    except FileNotFoundError:
      # doesn't exist, create it.
      self.mem = shared_memory.SharedMemory(name=self.shared_mem_name, create=True,
        size=total_sample_size * len(self.dataset))

      # initialize those single-byte flags with 0.
      # it's OK if this overwrites the progress of some concurrent processes; it just means a bit more loading overhead.
      self.mem.buf[::total_sample_size] = bytes(len(self.dataset))

  def __getitem__(self, index):
    """Return a single sample. Samples will be cached in the SharedMemory buffer
    and reused if possible.

    Args:
      index (int): Index of item.
    
    Returns:
      Tensor: The image.
    """
    if index >= len(self.dataset):
      raise StopIteration()

    if self.mem is None:  # may happen after serializing the dataset
      self._initialize()
    
    start_byte = self.sample_offsets[-1] * index

    if self.mem.buf[start_byte] == 0:
      # sample not loaded yet, load it from the wrapped Dataset
      sample = self.dataset[index]
      if torch.is_tensor(sample):
        sample = (sample,)  # wrap in tuple

      # cache each of the sample's tensors in the SharedMemory
      for i in range(len(self.sample_shapes)):
        start = start_byte + self.sample_offsets[i]
        end = start_byte + self.sample_offsets[i + 1]
        self.mem.buf[start:end] = sample[i].view(-1).numpy().tobytes()

      # finally, record that this sample has been loaded, in the first byte
      self.mem.buf[start_byte] = 1
    
    else:
      # the sample is cached, so extract it and convert to a tuple of tensors
      sample = []
      for i in range(len(self.sample_shapes)):
        start = start_byte + self.sample_offsets[i]
        end = start_byte + self.sample_offsets[i + 1]
        tensor = torch.frombuffer(self.mem.buf[start:end], dtype=self.sample_dtypes[i])
        sample.append(tensor.view(self.sample_shapes[i]))
      sample = tuple(sample)

    return sample

  def __len__(self):
    """Return the length of the dataset (number of samples). Defers to the wrapped Dataset.

    Returns:
      int: Number of samples.
    """
    return len(self.dataset)

  def __del__(self):
    """Close the SharedMemory handle on exit."""
    if hasattr(self, 'mem') and self.mem is not None:
      self.mem.close()

  def __getstate__(self):
    """Serialize without the SharedMemory references, for multiprocessing compatibility."""
    state = dict(self.__dict__)
    state['mem'] = None
    return state

  @classmethod
  def unlink(C, shared_mem_name):
    """Class method to unlink the SharedMemory buffer with a given name.
    Can be used in case the OS fails to release the buffer when it is not
    being used by any process. Should not be needed in normal operation."""
    try:
      mem = shared_memory.SharedMemory(name=shared_mem_name, create=False)
      mem.unlink()
      mem.close()
    except FileNotFoundError:
      print('SharedDataset: SharedMemory was not initialized, so unlinking has no effect.')


class TestDataset(Dataset):  # needs to be at the top level for DataLoader test
  """A dummy test dataset, returns 2 tensors with different dtypes and shapes"""
  def __init__(self):
    self.images = torch.rand(size=(5, 2))
    self.labels = torch.randint(size=(5,), high=10)
  def __getitem__(self, index):
    print('Wrapped dataset is loading sample', index)
    return (self.images[index,:], self.labels[index])
  def __len__(self):
    return self.labels.numel()


if __name__ == '__main__':
  # run some tests
  print('--- Iterate the wrapped dataset normally:')
  torch.manual_seed(0)
  dataset = TestDataset()
  for sample in dataset:
    print(sample)

  print('\n--- Iterate the shared dataset:\n(Should load samples from the wrapped dataset if this is the first process to do it. It always loads sample 0 at initialization.)')
  shared_dataset = SharedDataset(dataset, 'shared_dataset_test')
  for sample in shared_dataset:
    print(sample)

  print('\n--- Iterate the shared dataset again:\n(Will reuse cached values.)')
  for sample in shared_dataset:
    print(sample)

  print('\n--- Run DataLoader with 2 workers on shared dataset, batch size 2:')
  loader = torch.utils.data.DataLoader(shared_dataset,
    batch_size=2, num_workers=2, persistent_workers=True)
  for epoch in range(3):
    for (idx, (image, label)) in enumerate(loader):
      print("Batch", idx, "contains tensors of sizes", image.shape, "and", label.shape)

  input('\nRun this script again in another terminal to test the shared memory reuse.\nPress Enter to finish.\n(The shared memory will be deleted if this is the last script using it.)')

  # this is needed because the local tensors are reusing the SharedMemory's buffer.
  # if we don't delete the references to it before the SharedDataset is deleted,
  # it will not be able to close the SharedMemory buffer because it's still in use.
  # whether it happens or not depends on the order in which objects are deleted.
  del sample, image, label
