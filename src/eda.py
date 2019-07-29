import numpy as np
import pandas as pd
import librosa
import pickle
import seaborn as sns
from librosa.display import waveplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

x = np.array(librosa.core.load('../data/audio_train/Actor_05_speech/03-01-03-01-02-01-05.wav', sr=22050))[0]

fig, ax = plt.subplots()
plt.title('Happy WavePlot')
librosa.display.waveplot(x, sr=22050)
fig.savefig('Happy_waveplot')

y =  np.array(librosa.core.load('../data/audio_train/Actor_05_speech/03-01-05-01-02-01-05.wav', sr=22050))[0]

fig, ax = plt.subplots()
plt.title('Angry WavePlot')
librosa.display.waveplot(y, sr=22050)
fig.savefig('Angry_waveplot')


mfccx = librosa.feature.mfcc(y=x, sr=22050, n_mfcc=16)
mfccy = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=16)

fig, ax = plt.subplots(figsize=(5, 4))
plt.tight_layout()
plt.title('Angry MFCCs', fontsize=(20))
librosa.display.specshow(mfccx)
plt.show()
fig.savefig('angry_mfcc')

fig, ax = plt.subplots(figsize=(5, 4))
plt.tight_layout()
plt.title('Happy MFCCs', fontsize=(20))
librosa.display.specshow(mfccy)
plt.show()
fig.savefig('happy_mfcc')


def add_random_noise(self):
    noise_list = []
    for i in self.X:
        noise_amp = 0.005*np.random.uniform()*np.amax(i)
        noise = i.astype('float64') + noise_amp * np.random.normal(size=i.shape[0])
        noise_list.append(noise)
    if len(self.noise) == 0:
        self.noise = noise_list
        print('iteration')
    else:
        self.noise = self.noise + noise_list
        print('iteration')

noise_amp = 0.5*np.random.uniform()*np.amax(x)
noise = x.astype('float64') + noise_amp * np.random.normal(size=x.shape[0])

fig, ax = plt.subplots()
plt.title('White Noise WavePlot')
librosa.display.waveplot(noise, sr=22050)
fig.savefig('noise_waveplot_noise')


s_range = int(np.random.uniform(low=-5, high = 5)*20000)
shift = np.roll(x, s_range)
fig, ax = plt.subplots()
plt.title('Shifted WavePlot')
librosa.display.waveplot(shift, sr=22050)
fig.savefig('shift_waveplot_noise')

stretch = librosa.effects.time_stretch(x, 40)
fig, ax = plt.subplots()
plt.title('Stretched WavePlot')
librosa.display.waveplot(stretch, sr=22050)
fig.savefig('Stretch_waveplot_noise')
# libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D
# Get the iris dataset

sc = StandardScaler()
X_train = load_pickle('X_train_mfcc.pkl')
y_train = load_pickle('y_train_final.pkl')
labels = []
for i in y_train:
    if i == 2:
        labels.append('Calm')
    elif i == 3:
        labels.append('Happy')
    elif i == 4:
        labels.append('Sad')
    elif i == 5:
        labels.append('Angry')
X_train = [i.reshape(1024,) for i in X_train]
std_X = sc.fit_transform(X_train)
pca = PCA(3)
pca = pca.fit_transform(std_X)

fig, ax = plt.subplots(figsize=(10, 8))
# plt.scatter(pca[:, 0], pca[:, 1],
#             c=y_train, edgecolor='none', alpha=0.5)
sns.scatterplot(pca[:, 0], pca[:, 1], hue=labels)
plt.title('Principal Component Analysis for 4 Emotions', fontsize=20)
plt.xlabel('Principal Component 1', fontsize=18)
plt.ylabel('Principal Component 2', fontsize=18)
plt.legend()
fig.savefig('pca')
