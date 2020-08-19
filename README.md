# EmergentLanguage
Repository for people who are interested at topics related to Emergent Language

## Packages

 1. **data_loaders**: pre-process various kinds of inputs and aggregate them into torch.utils.data.DataLoader
  * data_loaders: functions that could return DataLoader objects
  * data_transforms: callable classes that could transform the original inputs to formats in PyTorch
  * data_sets: child class of torch.utils.data.Dataset that could indexing samples
  * *_local_unit_test*: functions to test the correctness of classes/functions in the above three modules
  

2. **vaes**: collections of various disentangling VAEs, in which the encoders will be re-used in referential games.

3. **agents**: implementation of agents in the games including speakers and listeners.
  * encoders: child class of torch.nn.modules that could encode input objects in different formats, e.g. ImgEncoderCNN (CNN-based encoders for images).
  * decoders: child class of torch.nn.modules that could generate objects in different formats, e.g. SeqDecoderLSTM (LSTM-based decoder to generate sequences).
  * speakers: child class of torch.nn.modules that take objects in different formats as inputs and output sequences containing discrete symbols, i.e. an encoder (e.g. ImgEncoderCNN) + a decoder (SeqDecoderLSTM).
  * listeners: child class of torch.nn.modules that takes sequences as input and encode them into representation vectors, i.e. an encoder for sequences (e.g. SeqEncoderLSTM). **Note** that the further actions of listeners (e.g. choice in referential games) are implemented by different sub-classes listed as follows:
    * ReferListener: listeners that are specifically designed for referential game, the output are cosine similarity between embeddings for different candidates and the input message.
    * ReconstructListener: listeners that are specifically designed for reconstruction game, the output are a reconstructed object (in the same format of the input object).

4. **games**: implementation of different kinds of games which are actually global trainers in canonical deep learning projects that control: i) data loading; ii) agents interacting; iii) parameters update based on rewards/losses.