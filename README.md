# Epigenetic_Clocks
Hello! Welcome to the world of epigenetic clocks and age prediction.

Here, me and my teammates implemented and evaluated 2 transformer encoder architectures for age inference based on the DNA methylation values. 
We also reproduced the previous results using the multilayer perceptron (Camillo et.al.) and the linear regression (Horvath et.al.)

In this repository, you will find 3 folders.
The first folder, "Regressor and dataset in R" contains R markdown notebook with penalized linear regression implementation as well as the code for collecting the dataset.
The second folder, "Dataset construction" contains a Python notebook with some hybrid cells that use R for dataset normalization
The third folder, "Models", which is probably the most interesting, contains the MLP as well as multiple transformer encoder models.

The pre-trained models and the pre-processed datasets are available via the public link below:
https://drive.google.com/drive/folders/11JlLkkm6oNcmSivNUq0Nyb25hPNxg45M?usp=share_link

Note that the link has 2 dataset files, "large_dataset.pkl" and "best_dataframe.csv" are essentially the same, but the names are different to make them compatible with the code.

If you have any questions, feel free to reach out (winterblizzard19@gmail.com)

