# BirdSound_Classifier

### Problem:
The Problem is to use your machine-learning skills to identify under-studied [Indian bird species](https://stateofindiasbirds.in/) by sound. Specifically, the challenge was to develop computational solutions to process continuous audio data and recognize the species by their calls.

### Data:

You can download the dataset from Kaggle Link: [BirdClef 2024 DataSet](https://www.kaggle.com/competitions/birdclef-2024/data) and can understand the features and
sounds.

### Solution:
1) The Pre-processing like conversion from audio to mel-spectrograms, applying transformation techniques on mel-spectrograms (GuassianNoise, PinkNoise, NoiseInjection, etc) and we also used the center 5-secs of the audio, but these steps take too much time.  To speed up the audio-to-spectrogram, we employ CuPy. CuPy is a NumPy/SciPy-compatible array library for GPU-accelerated computing with Python, which can significantly improve conversion efficiency.

2) After that, Some Augmentation like Flipping and XYMasking are applied to the pre-processed data. 

3) At Last, created a baseline of EfficientNetB0 with PyTorch. additionally, PyTorch-Lightning is employed to organize the training. AUC Metric used with 5-KFolds.
