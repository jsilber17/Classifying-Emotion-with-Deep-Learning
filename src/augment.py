import numpy as np
import librosa
import pickle

class Augment():

    def __init__(self, X_train_filepath, X_test_filepath, X_holdout_filepath, y_train_filepath):
        ''' Load in X and y data for augmentation'''

        self.X_train_filepath = X_train_filepath
        self.X_test_filepath = X_test_filepath
        self.X_holdout_filepath = X_holdout_filepath
        self.y_train_filepath = y_train_filepath
        self.X_train = list(np.load(self.X_train_filepath, allow_pickle=True))
        self.X_test = list(np.load(self.X_test_filepath, allow_pickle=True))
        self.X_holdout = list(np.load(self.X_holdout_filepath, allow_pickle=True))
        self.y_train = list(np.load(self.y_train_filepath, allow_pickle=True))
        self.noise = []
        self.stretch = []
        self.shift = []

    def add_random_noise(self):
        ''' Add random noise to the .wav file
            This function doubles the amount of original training data '''

        noise_list = []
        for i in self.X_train:

            # Multipy a random number between 0 and 2 by the maximum value in the audio array
            noise_amp = 2*np.random.uniform()*np.amax(i)

            # Add the random number above to the audio array and multiply the array by a normal distribution
            noise = i.astype('float64') + noise_amp * np.random.normal(size=i.shape[0])

            noise_list.append(noise)
        if len(self.noise) == 0:
            self.noise = noise_list
        else:
            self.noise = self.noise + noise_list

    def stretch_array(self, rate):
        ''' Randomly stretch or shrink the time array
            This function doubles the amount of original training data '''

        stretch = [librosa.effects.time_stretch(audio, rate) for audio in self.X_train]
        if len(self.stretch) == 0:
            self.stretch = stretch
        else:
            self.stretch = self.stretch + stretch

    def shift_data(self):
        ''' Randomly shift the time array on the X axis
            This function doubles the amount of original training data '''

        s_range = int(np.random.uniform(low=-5, high = 5)*500)

        # Shifts the array by a random number defined above
        shift = [np.roll(audio, s_range) for audio in self.X_train]

        if len(self.shift) == 0:
            self.shift = shift
        else:
            self.shift = self.shift + shift

    def mean(self, data):
        return np.mean(data)

    def standard_deviation(self, data):
        return np.std(data)

    def get_mfccs(self, arr):
        ''' Return 16 Mel Frequency Cepstral Coefficients for each audio file in X data '''

        mfcc = [librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=16) for audio in arr]
        return mfcc

    def resize_mfccs(self, arr):
        ''' Fix the size of the MFCCs to get a uniform value for each MFCC '''

        self.target_size = 64
        re_mfcc = [librosa.util.fix_length(mfcc, self.target_size, axis=1) for mfcc in arr]
        return re_mfcc

    def standardize_data(self, arr):
        std_data = arr - self.mean(arr)/self.standard_deviation(arr)
        return arr

    def load_pickle(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    def dump_pickle(self, filename, arr):
        with open(filename, 'wb') as f:
            pickle.dump(arr, f)

    def finalize_X_data(self, noise_iterations, shift_iterations, stretch_iteration):
        ''' Augment the data, get, resize, and standardize the MFCCs then dump them in a directory '''

        for i in range(noise_iterations):
            self.add_random_noise(7)
        for i in range(shift_iterations):
            self.shift_data(2)
        for i in range(stretch_iterations):
            self.stretch_array(2)

        self.X_train = self.X_train + self.noise + self.stretch + self.shift
        self.X_train = self.get_mfccs(self.X_train)
        self.X_train = self.resize_mfccs(self.X_train)
        self.X_train = self.standardize_data(self.X_train)

        self.X_test = self.get_mfccs(self.X_test)
        self.X_test = self.resize_mfccs(self.X_test)
        self.X_test = self.standardize_data(self.X_test)

        self.X_holdout = self.get_mfccs(self.X_holdout)
        self.X_holdout = self.resize_mfccs(self.X_holdout)
        self.X_holdout = self.standardize_data(self.X_holdout)

        self.dump_pickle('X_train_mfcc.pkl', self.X_train)
        self.dump_pickle('X_test_mfcc.pkl', self.X_test)
        self.dump_pickle('X_holdout_mfcc.pkl', self.X_holdout)

    def finalize_y_data(self):
        ''' Create a new y_train to match the size of the augmented X_train '''

        mult = len(self.X_train) // len(self.y_train)
        self.y_train = self.y_train * mult
        self.dump_pickle('y_train_final.pkl', self.y_train)

def main():
    ad = Augment('X_train', 'X_test', 'X_holdout', 'y_train')
    ad.finalize_X_data(7, 2, 2)
    ad.finalize_y_data()

if __name__ == '__main__':
    main()
