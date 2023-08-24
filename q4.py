import random
from scipy.io import wavfile
from scipy.signal import find_peaks
import matplotlib.pyplot as plot
import numpy as np
from sklearn.decomposition import FastICA, PCA
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import linear_model
from scipy import signal
#from python_speech_features import mfcc

#------------------------------------------------------------------------------
# reads the wav file
def readFile(filename):
    samplingFrequency, signalData = wavfile.read(filename)
    sampling_frequencies.append(samplingFrequency)
    signal_datas.append(signalData)

# draws the specograms
def spectograms(signals, original, labels):
    for index, signal_data in enumerate(signals):
        # Plot the signal read from wav file    
        plot.subplot(211)
        if original:
            plot.title('Spectrogram of a wav file {}'.format(file_names[index]))
        else:
            plot.title('ICA estimated source{}'.format(labels[index]))
        plot.plot(signal_data)
        plot.xlabel('Sample')
        plot.ylabel('Amplitude')
        plot.subplot(212)
        plot.specgram(signal_data,Fs=sampling_frequencies[index])
        plot.xlabel('Time')
        plot.ylabel('Frequency')
        plot.show()

# return the mixing matrix
def getMixingMatrix():
    mixing_matrix = []
    for i in range(0,6):
        row_matrix = []
        for i in range(0,6):
            row_matrix.append(random.uniform(0.5, 2.5))
        mixing_matrix.append(row_matrix)
    return mixing_matrix

#------------------------------------------------------------------------------
# plots the feature
def make_plt(title, xlabel,ylabel, y, x):
    colors = []
    for label in y:
        if label == 1:
            colors.append('red')
        else:
            colors.append('blue')
    _, ax = plot.subplots()
    plot.scatter(x, y, c=colors)
    plot.ylabel(ylabel)
    plot.xlabel(xlabel)
    plot.title(title)
    plot.show()

# Feature that returns how many amplitudes are more than the average amplitude
# of the signal
def countMoreThanAvg(signal_data):
    avg = np.average(signal_data)
    return np.count_nonzero(signal_data > avg)

# Feature that returns how many peaks in the signal
def countPeaks(signal_data):
    peaks, _ = find_peaks(signal_data, height=0)
    return len(peaks)

# Feature that counts the varience of the signal
def calcVarience(signal_data):
    return np.var(signal_data)

# Feature that compare parts of the singal (each second). Counts if the sum
# of the amplitudes of the second is greater than the following seconds.
# return how many times the condition is met.
def comparePartsOfWav(signal_data, frequency):
    data_for_each_sec = []
    time = int(len(signal_data)/frequency)
    for i in range(time):
        data_for_each_sec.append(signal_data[i*frequency:(i+1)*frequency])
    counter = 0
    for i in range(len(data_for_each_sec)):
        for j in range(i+1, len(data_for_each_sec)):
            if sum(data_for_each_sec[i]) > sum(data_for_each_sec[j]):
                counter+=1
    return counter

#-----------------------------------main---------------------------------------
# A
file_names = ['source1.wav', 'source2.wav', 'source3.wav', 
              'source4.wav', 'source5.wav', 'source6.wav']
sampling_frequencies = []
signal_datas = []

for file_name in file_names:
    readFile(file_name)        

# B
spectograms(signal_datas, True, [])

# C
mixing_matrix = np.array(getMixingMatrix())

# D
signal_sumation = np.array(signal_datas).T
X = np.dot(signal_sumation, mixing_matrix.T)
X_T = X.T
for frquency, i in zip(sampling_frequencies, range(len(X_T))):
    wavfile.write("mixedSource{}.wav".format(i+1), frquency, X_T[i].astype(np.int16))

# E
reconstructed_signals = []
ica = FastICA(whiten="arbitrary-variance")
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix
# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)
# For comparison, compute PCA
pca = PCA(n_components=6)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components
#spectograms(H.T, False, ['']*6)

# F
S_T = S_.T
for frquency, i in zip(sampling_frequencies, range(len(S_T))):
    wavfile.write("reconstructedSource{}.wav".format(i+1), frquency, S_T[i].astype(np.int16))

# G
"""
The reconstructrd signals are not in order. For determining which signal is which
first we print the spectograms, and in line 232 we manually decide which 
signal is corrollated to the original - (The comment in lines 203 - 208)
"""
spectograms(S_T, False, ['']*6) 


#------------------------------------------------------------------------------
# returns the arrays of the features
# H, I
def getFeatures(signal_datas):
    averages_feature = []
    for signal_data in signal_datas:
        averages_feature.append(countMoreThanAvg(signal_data))
    make_plt('Feature 1: Amplitudes more than average', 
             'Amplitudes more than average', 'lables 0-1', audio_labels, averages_feature)
    
    peaks_feature = []
    for signal_data in signal_datas:
        peaks_feature.append(countPeaks(signal_data))
    make_plt('Feature 2: Peaks of signals', 
             'Peaks of signals', 'lables 0-1', audio_labels, peaks_feature)
    
    variance_feature = []
    for signal_data in signal_datas:
        variance_feature.append(calcVarience(signal_data))
    make_plt('Feature 3: Variances of wavs', 
             'Variances of wavs', 'lables 0-1', audio_labels, variance_feature)
    
    compare_parts_feature = []
    for signal_data in signal_datas:
        compare_parts_feature.append(comparePartsOfWav(signal_data, 8000))
    make_plt('Feature 4: Compare parts of wavs', 
             'Diffrences of parts of audio', 'lables 0-1', audio_labels, compare_parts_feature)
    
    return averages_feature, peaks_feature, variance_feature, compare_parts_feature


#------------------------------------------------------------------------------
"""
0 - non speaking
1 - speaking
"""
audio_labels = np.array([0, 1, 1, 1, 0, 1])

#get Features
averages_feature, peaks_feature, variance_feature, compare_parts_feature = getFeatures(signal_datas)

# creating the X (feature) matrix
X = np.column_stack((averages_feature, peaks_feature, variance_feature, compare_parts_feature))
X_scaled = preprocessing.scale(X)

# the predicted outputs
Y = audio_labels

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.5, shuffle=False)

# Training Logistic regression
logistic_classifier = linear_model.LogisticRegression(solver='lbfgs')
logistic_classifier.fit(X_train, y_train)

# show how good is the classifier on the training data
expected = Y
predicted = logistic_classifier.predict(X_test)

classification = []
for prediction, y in zip(predicted, y_test):
    classification.append((prediction, y))
    
print("Logistic regression using [countMoreThanAvg, countPeaks, calcVarience, comparePartsOfWav] features:\n%s\n" % (
 metrics.classification_report(
 y_test,
 predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted))


#------------------------------------------------------------------------------
# J
averages_feature_R, peaks_feature_R, variance_feature_R, compare_parts_feature_R = getFeatures(S_T)

# creating the X (feature) matrix
X = np.column_stack((averages_feature_R, peaks_feature_R, variance_feature_R, compare_parts_feature_R))
X_scaled = preprocessing.scale(X)
reconstructed_signal_labels = np.zeros(6)
original_indexes_files = []

"""
This is not a comment, this is where we manually determine which reconstructed signal
spectogram is matched to the original. We made it a comment so that the program will run
without disturbance.
for index, value in enumerate(reconstructed_signal_labels):
   index_of_original = int(input("enter the original: ")) - 1     
   reconstructed_signal_labels[index] = audio_labels[index_of_original]
   original_indexes_files.append(index_of_original+1)
"""

for index, original_signal in enumerate(signal_datas):
    similarities = []
    for recontructed_signal in S_T:
        # find the corr beteween the original signal and the reconstructed  
        #euclidean_dist = np.linalg.norm(original_signal - recontructed_signal)
        # corr = np.corrcoef(original_signal, recontructed_signal)[0,1]   
        corr = signal.correlate(original_signal, recontructed_signal, mode='same')
        similarities.append(corr)
    min_sig = np.argmin(similarities)
    min_sig = int(min_sig/50000)
    reconstructed_signal_labels[min_sig] = audio_labels[index]
    original_indexes_files.append(min_sig+1)

spectograms(S_T, False, original_indexes_files)   

predicted = logistic_classifier.predict(X_scaled)

print("Logistic regression using [countMoreThanAvg, countPeaks, calcVarience, comparePartsOfWav] features:\n%s\n" % (
 metrics.classification_report(
 reconstructed_signal_labels,
 predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(reconstructed_signal_labels, predicted))

