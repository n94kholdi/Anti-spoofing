from AntiSpoofing import AntiSpoof

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
# from tensorflow_core.python.keras.models import load_model

from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

## Choose your analysis method:
method = input("""Select number of your analysis method:\n
           1. Classification_KNN
           2. Classification_SVM
           3. Anomaly Detection_SVM
           4. Anomaly Detection_NeuralNet\n
method: """)

pca = input(""" Do you want PCA for preprocessing:
           1. Yes
           2. NO
pca: """)
if pca == '1':
    pca = True
else:
    pca = False


### Path of Train and Test files:
Train_path = '../dataset/train/'
Test_path = '../dataset/test/'

AntiSpoof = AntiSpoof(method)
AntiSpoof.Extract_Features(Train_path, 'Train')

### Split real data from whole of dataset for Anomaly Detection
RealData = AntiSpoof.Data[150:]
### Shuffle DataSet
AntiSpoof.Data, AntiSpoof.Labels = shuffle(AntiSpoof.Data, AntiSpoof.Labels, random_state=0)
AntiSpoof.Extract_Features(Test_path, 'Test')



if method == '1':
    ###############################
    ### HyperParameter Tunning: ###

    kf = KFold(n_splits=5)

    print('k          Validation Accuracy\n---------------------------')
    for k in range(1, 10, 2):
        avg_acc = 0

        ### Using Kfold for Cross-validation:
        for train_index, valid_index in kf.split(AntiSpoof.Data):
            ### Split DataSet to train and validation
            AntiSpoof.train_valid_split(train_index, valid_index)
            ### Using PCA for Dimensional Reduction
            if pca == True:
                AntiSpoof.PCA()
            ### Using Knn as classifier
            acc, _ = AntiSpoof.KNN(k, 'Train')
            avg_acc += acc

        print(k, '   |  ', avg_acc / 5)

    ### Preprocess Test Data:
    AntiSpoof.X_test = AntiSpoof.preprocess(AntiSpoof.X_test, pca)

    ### Evaluate model on test data:
    Test_acc, pred_test = AntiSpoof.KNN(1, 'Test')
    print('\nTest Accuracy :', Test_acc)

    ### Compute confusion matrix:
    print('\nPlot Confusion matrix and rate analysis:')
    AntiSpoof.ConfusionMat(AntiSpoof.y_test, pred_test)

    ### Plot Samples:
    AntiSpoof.plot_samples(Test_path, pred_test)


elif method == '2':

    kf = KFold(n_splits=5)
    avg_acc = 0

    ### Using Kfold for Cross-validation:
    print('Use polynomial kernel function:\n')
    print('degree          Validation Accuracy\n-----------------------------------')
    best_svm = 0
    best_acc = -100
    for d in range(1, 6):
        avg_acc = 0
        for train_index, valid_index in kf.split(AntiSpoof.Data):
            ### Split DataSet to train and validation
            AntiSpoof.train_valid_split(train_index, valid_index)

            ### Using PCA for Dimensional Reduction
            if pca == True:
                AntiSpoof.PCA()

            ### Using SVM as Classifier
            svm = make_pipeline(StandardScaler(), SVC(gamma='scale', kernel='poly', degree=d))
            svm.fit(AntiSpoof.X_train, AntiSpoof.y_train)

            ### Compute Accuracy of model on validation set
            pred_valid = svm.predict(AntiSpoof.X_valid)
            acc = accuracy_score(AntiSpoof.y_valid, pred_valid)

            avg_acc += acc

        ## Find best degree of polynomial kernel
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_svm = svm

        print(d, '   |\t', avg_acc / 5)

    ### Preprocess Test Data:
    AntiSpoof.X_test = AntiSpoof.preprocess(AntiSpoof.X_test, pca)

    ### Evaluate model on test data:
    svm = best_svm
    print(svm)
    pred_test = svm.predict(AntiSpoof.X_test)
    Test_acc = accuracy_score(AntiSpoof.y_test, pred_test)
    print('\nTest Accuracy :', Test_acc)

    # ### Compute confusion matrix:
    print('\nPlot Confusion matrix and rate analysis:')
    AntiSpoof.ConfusionMat(AntiSpoof.y_test, pred_test)

    ### Plot Samples:
    AntiSpoof.plot_samples(Test_path, pred_test)

elif method == '3':

    Models = ['one-class-SVM', 'Isolation Forest']
    ### Train Anomaly Detection model on RealData:
    model = AntiSpoof.AnomalyDetector(RealData, pca, Models[0])

    ### Evaluate model on X_test:
    AntiSpoof.X_test = AntiSpoof.preprocess(AntiSpoof.X_test, pca)
    anomaly_pred = model.predict(AntiSpoof.X_test)

    ### (0 = real) and (1 = attack)
    anomaly_pred = [0 if (p == 1) else 1 for p in anomaly_pred]

    ### Evaluate model on Test Data:
    AntiSpoof.Evaluate_model(anomaly_pred)

    ### Plot Samples:
    AntiSpoof.plot_samples(Test_path, anomaly_pred)

elif method == '4':

    kf = KFold(n_splits=5)

    for train_index, valid_index in kf.split(AntiSpoof.Data):

        ### Split DataSet to train and validation
        AntiSpoof.train_valid_split(train_index, valid_index)

        ### Using PCA for Dimensional Reduction
        if pca == True:
            AntiSpoof.PCA()

        ### Train Anomaly Detection model on RealData:
        num_epochs = 20
        model = AntiSpoof.AnomalyDetector(RealData, pca, 'AutoEncoder', num_epochs)

        ### Predict X_test by model:
        AntiSpoof.X_test = AntiSpoof.preprocess(AntiSpoof.X_test, pca)
        AutoEnc_predictions = model.predict(AntiSpoof.X_test)

        mse = np.mean(np.power(AntiSpoof.X_test - AutoEnc_predictions, 2), axis=1)
        error_df = pd.DataFrame({'reconstruction_error': mse,
                                 'true_class': AntiSpoof.y_test})

        groups = error_df.groupby('true_class')
        fig, ax = plt.subplots()

        threshold = 200
        for name, group in groups:
            ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
                    label="Attack" if name == 1 else "Real")
        ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
        ax.legend()
        plt.title("Reconstruction error for different classes")
        plt.ylabel("Reconstruction error")
        plt.xlabel("Data point index")
        plt.show();
        # print(error_df.describe())

        anomaly_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]

        ### Evaluate model on Test Data:
        AntiSpoof.Evaluate_model(anomaly_pred)

        ### Plot Samples:
        AntiSpoof.plot_samples(Test_path, anomaly_pred)

        break
