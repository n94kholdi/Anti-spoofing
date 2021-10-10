# import the necessary packages
import numpy as np
import seaborn as sns
from skimage.filters import difference_of_gaussians
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import regularizers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from matplotlib import pyplot as plt, image as mpimg
from scipy import stats
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from tensorflow_core.python.keras.models import load_model

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from skimage import feature
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
import cv2


class AntiSpoof:

        def __init__(self, method):

            self.Data = []
            self.Labels = []
            self.X_test = []
            self.y_test = []
            self.File_names_test = []

            self.pca = PCA(svd_solver='full')
            self.scaler = StandardScaler()
            self.method = method

        def LBP(self, image, numPoints, radius, eps=1e-7):
            # compute the Local Binary Pattern representation
            # of the image, and then use the LBP representation
            # lbp = feature.local_binary_pattern(image, numPoints, radius, method="uniform")
            lbp = difference_of_gaussians(image, 2, 10)
            lbp = lbp.astype("float")
            lbp /= (lbp.sum() + eps)


            return lbp

        def Extract_Features(self, Data_path, mode):

            # initialize the local binary patterns descriptor along with
            # the data and label lists

            Data = []
            Labels = []

            File_names = []
            if mode == 'Train':
                for i in range(0, 150):
                    File_names.append(str(i)+'.jpg')
            if mode == 'Test':
                    for i in range(150, 200):
                        File_names.append(str(i)+'.jpg')
                    self.File_names_test = File_names

            for img in File_names:

                ### Attack files:
                image = cv2.imread(Data_path + 'attack/' + img)
                # image = cv2.imread(img)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = gray
                if self.method == '2':
                    image = self.LBP(gray, 10, 30)
                # lbp_image = self.LBP(gray, 24, 8)

                Data.append(image)
                Labels.append(1)

            for img in File_names:
                ### Real files:
                image = cv2.imread(Data_path + 'real/' + img)
                # image = cv2.imread(img)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = gray
                if self.method == '2':
                    image = self.LBP(gray, 10, 30)
                # lbp_image = self.LBP(gray, 24, 8)

                Data.append(image)
                Labels.append(0)

            if mode == 'Train':
                self.Data = np.array(Data)
                self.Labels = np.array(Labels)
                return (self.Data, self.Labels)
            else:
                self.X_test = np.array(Data)
                self.y_test = np.array(Labels)
                a = 0

        def train_valid_split(self, train_index, test_index):

            self.X_train, self.X_valid = self.Data[train_index], self.Data[test_index]
            self.y_train, self.y_valid = self.Labels[train_index], self.Labels[test_index]

            self.X_train = self.X_train.reshape(self.X_train.shape[0], -1)
            self.X_valid = self.X_valid.reshape(self.X_valid.shape[0], -1)

        def PCA(self):

            # self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_train = self.pca.fit_transform(self.X_train)

            # self.X_valid = self.scaler.transform(self.X_valid)
            self.X_valid = self.pca.transform(self.X_valid)


        def KNN(self, k, mode):

            if mode == 'Train':
                X = self.X_valid
                Y = self.y_valid
            else:
                X = self.X_test
                Y = self.y_test

            pred = []
            for (img, label) in zip(X, Y):

                neighb_dist = np.ones(k) * 10000000
                neighb_labels = np.ones(k) * -1

                for ind in range(0, len(self.X_train)):

                    img_train = self.X_train[ind]
                    ## Euclidean distance:
                    dist = np.sqrt(np.sum((img - img_train) ** 2))

                    if dist < np.max(neighb_dist):
                        ind_max = np.argmax(neighb_dist)
                        neighb_dist[ind_max] = dist
                        ## Store labels of nearest neighbors
                        neighb_labels[ind_max] = self.y_train[ind]

                chosen_label = stats.mode(neighb_labels, axis=None)[0][0]
                pred.append(chosen_label)

            acc_score = accuracy_score(Y, pred)
            # print(k, ':', acc_score_valid)
            return acc_score, pred

        def preprocess(self, Data, pca, mode='Test'):

            X = Data
            X = X.reshape(X.shape[0], -1)
            if pca == True:

                if mode == 'Train':
                    X = self.scaler.fit_transform(X)
                    X = self.pca.fit_transform(X)
                else:
                    # X = normalize(X)
                    try:
                        X = self.scaler.transform(X)
                    except:
                        pass
                    X = self.pca.transform(X)

            return X

        def AnomalyDetector(self, RealData, pca, model_name='one-class-SVM', NN_epochs=100):

            RealData = RealData.reshape(RealData.shape[0], -1)
            RealData = self.scaler.fit_transform(RealData)
            if pca == True:
                RealData = self.pca.fit_transform(RealData)

            if model_name == 'one-class-SVM':

                model = svm.OneClassSVM(nu=.2, kernel='linear', gamma=.001, degree=3)
                model.fit(RealData)

                return model

            elif model_name == 'Isolation Forest':

                rs = np.random.RandomState(0)
                model = IsolationForest(max_samples=10, random_state=rs, contamination=.1)
                model.fit(RealData)

                return model

            elif model_name == 'AutoEncoder':

                RealData = RealData.reshape(RealData.shape[0], -1)

                # model = load_model('AutoEnc_model.h5')
                # return model

                input_dim = RealData.shape[1]
                encoding_dim = 100

                input_layer = Input(shape=(input_dim,))
                # encoder = Dense(encoding_dim, activation="tanh",
                #                 activity_regularizer=regularizers.l1(10e-5))(input_layer)
                encoder = Dense(int(encoding_dim / 2), activation="relu")(input_layer)
                decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
                decoder = Dense(input_dim, activation='relu')(decoder)
                autoencoder = Model(inputs=input_layer, outputs=decoder)

                nb_epoch = NN_epochs
                batch_size = 16
                autoencoder.compile(optimizer='adam',
                                    loss='mean_squared_error',
                                    metrics=['accuracy'])
                checkpointer = ModelCheckpoint(filepath="AutoEnc_model.h5",
                                               verbose=0,
                                               save_best_only=True)

                history = autoencoder.fit(RealData, RealData,
                                          epochs=nb_epoch,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          validation_split=0.15,
                                          verbose=1,
                                          callbacks=[checkpointer]).history

                plt.plot(history['accuracy'])
                plt.plot(history['val_accuracy'])
                plt.title('model acc')
                plt.ylabel('acc')
                plt.xlabel('epoch')
                plt.legend(['train', 'valid'], loc='upper right');
                plt.show()

                plt.plot(history['loss'])
                plt.plot(history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'valid'], loc='upper right');
                plt.show()

                return autoencoder

        def ConfusionMat(self, y_test, y_pred):

            conf_mat = confusion_matrix(y_test, y_pred)
            self.plot_confusionMat(conf_mat)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[1, 0]).ravel()
            print('\nTrue Positive:  %f\nFalse Positive: %f\nTrue Negative:  %f\nFalse Negative: %f' % (tp, fp, tn, fn))

        def plot_confusionMat(self, conf_mat):

            LABELS = ["Real", "Attack"]
            plt.figure(figsize=(12, 12))
            sns.heatmap(conf_mat, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
            plt.title("Confusion matrix")
            plt.ylabel('True class')
            plt.xlabel('Predicted class')
            plt.show()

        def Evaluate_model(self, anomaly_pred):

            ### Accuracy of model on test data:
            acc_score = accuracy_score(self.y_test, anomaly_pred)
            print('Test Accuracy : ', acc_score)

            ### Compute confusion matrix:
            print('\nPlot Confusion matrix and rate analysis:')
            self.ConfusionMat(self.y_test, anomaly_pred)

        def plot_samples(self, Data_path, preds):


            fig, axs = plt.subplots(3, 5)
            fig.suptitle('Images of Attack DataSet')
            i = 0
            for (img, label, pred) in zip(self.File_names_test[0:15], self.y_test[0:15], preds):

                ### Attack files:
                image = mpimg.imread(Data_path + 'attack/' + img)
                axs[int(i / 5), i % 5].imshow(image)
                axs[int(i / 5), i % 5].set_ylabel(img)
                axs[int(i / 5), i % 5].set_xlabel('true_label : %i, pred : %i' % (label, pred))
                i += 1
            plt.show()

            fig, axs = plt.subplots(3, 5)
            fig.suptitle('Images of Real DataSet')
            i = 0
            for (img, label, pred) in zip(self.File_names_test[0:15], self.y_test[50:65], preds[50:65]):
                ### Attack files:
                image = mpimg.imread(Data_path + 'real/' + img)
                axs[int(i / 5), i % 5].imshow(image)
                axs[int(i / 5), i % 5].set_ylabel(img)
                axs[int(i / 5), i % 5].set_xlabel('true_label : %i, pred : %i' % (label, pred))
                i += 1
            plt.show()

