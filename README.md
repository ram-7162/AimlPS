(1)  How i process to final code is as follows:
        Initially colab could no able to find folder path so code copies the ZIP into colab then extract it inside colab then automatically updates path for train/test and sample sub.
        Length of audio files mismatch so code crop/pad to make them of 4 sec length.
        Convert Audio using Log-mel spectrogram : Each sample becomes a 2D image-like representation


(2) model in code:
        model use is CNN
        optimizer : Adam
        loss function : Sparse categorical crossentropy



One of callback used is EarlyStopping(to avoid overfitting).