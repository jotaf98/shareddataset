# SharedDataset

A PyTorch `Dataset` that keeps a cache of arbitrary tensors in shared memory, accessible globally to all processes.

This can yield enormous memory savings if you have multiple processes that access the same dataset (such as parallel training runs or `DataLoader` workers).


**Why?** Keeping a dataset in memory (e.g. with PyTorch's `TensorDataset`) is much faster than reading it from disk. This is feasible for many medium-sized datasets (e.g. RGB images will take `3*width*height*number_of_images/1024**3` GB). However, this is multiplied by the number of processes holding that dataset, often going over the memory limit. `SharedDataset` allows all processes to share the same memory, reusing the same copy.

**How?** `SharedDataset` simply wraps another dataset (for example, one that loads images from disk), and only calls it the first time that a sample is accessed. These values are cached using Python's `SharedMemory`, and retrieved later. So the first pass over the data may be slow, but afterwards each sample is loaded instantly. The shared buffer is deallocated automatically when the last process is done.


## Example

Using TorchVision's `ImageDataset` as an example (not required in general):

```python
from shareddataset import SharedDataset
from torchvision.datasets import ImageDataset

# a slow-loading dataset (could be any arbitrary Dataset instance)
my_dataset = ImageDataset('/data/myimages/')

# the shared dataset cache -- the second argument is a unique name
shared_dataset = SharedDataset(my_dataset, 'my_dataset')

# first pass over the data, reads files (slow) but caches the result
for (image, label) in shared_dataset:
  print(image.shape, label)

# second pass over the data, no files are read (fast)
for (image, label) in shared_dataset:
  print(image.shape, label)

# if you stop the script here, and rerun it in another console, it
# will reuse the cache, which is also fast
input()
```

With DataLoaders instead:
```python
# the worker processes of a DataLoader all share the same memory.
# use persistent workers to ensure the SharedDataset is not deallocated
# between epochs.
loader = torch.utils.data.DataLoader(shared_dataset,
  batch_size=100, num_workers=4, persistent_workers=True)
for epoch in range(3):
  for (image_batch, labels) in loader:
    print(image_batch.shape, labels)
```

You can also run `shareddataset.py` as a script to run a similar, self-contained test (without image files).


## Author

[João Henriques](http://www.robots.ox.ac.uk/~joao/), [Visual Geometry Group (VGG)](http://www.robots.ox.ac.uk/~vgg/), University of Oxford
