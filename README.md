# EmergentLanguage
Repository for people who are interested at topics related to Emergent Language

## Packages

 1. **data_loaders**: pre-process various kinds of inputs and aggregate them into torch.utils.data.DataLoader
  - data_loaders: functions that could return DataLoader objects
  - data_transforms: callable classes that could transform the original inputs to formats in PyTorch
  - data_sets: child class of torch.utils.data.Dataset that could indexing samples
  - *_local_unit_test*: functions to test the correctness of classes/functions in the above three modules
  

2. **vaes**: collections of various disentangling VAEs, in which the encoders will be re-used in referential games.