import numpy as np
import librosa
import os, sys
import glob
import pickle


class Data():

    def __init__(self, train_directory, test_directory, holdout_directory):
        ''' Load in directory names containing the train, test, and holdout audio files '''

        self.train_directory = glob.glob(train_directory)
        self.test_directory = glob.glob(test_directory)
        self.holdout_directory = glob.glob(holdout_directory)

    def dump_pickle(self, filename, arr):
        ''' Dump a pickle file into a directory '''

        with open(filename, 'wb') as f:
            pickle.dump(arr, f)

    def load_pickle(self, filename):
        ''' Load in a pickle file from a directory '''

        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    def get_audio_files(self, directory, X_filename, y_filename):
        ''' Load in all of the .wav files from a directory and their emotion labels and create X and y lists
            Dumps X and y pickle files in the same directory as the script '''

        X = []
        y = []

        for f in directory:
            # Iterating through all the files in a directory
            for file in os.listdir(f):
                try:
                    files = f + '/' + file

                    # Load the file from a directory on local machine  -- audio file starts at 0.5 seconds and continues for 3 seconds
                    array = librosa.core.load(files, res_type='kaiser_fast', duration=3, offset=0.5)[0]
                    
                    # Emotion is defined in the filename (at position 7:8)
                    if int(file[7:8]) == 2: # Calm
                        emotion = int(file[7:8])
                        X.append(array)
                        y.append(emotion)
                    elif int(file[7:8]) == 3: # Happy
                        emotion = int(file[7:8])
                        X.append(array)
                        y.append(emotion)
                    elif int(file[7:8]) == 4: # Sad
                        emotion = int(file[7:8])
                        X.append(array)
                        y.append(emotion)
                    elif int(file[7:8]) == 5: # Angry
                        emotion = int(file[7:8])
                        X.append(array)
                        y.append(emotion)
                    else:
                        pass
                except:
                    pass
        self.X, self.y = X, y
        print('Dumping pickle...')
        self.dump_pickle(X_filename, self.X)
        self.dump_pickle(y_filename, self.y)
        print('Pickle dumped!')


def main():
    data = Data('../data/audio_train/*', '../data/audio_test/*', '../data/audio_holdout/*')
    print('Getting train audio files...')
    data.get_audio_files(data.train_directory, 'X_train', 'y_train')
    print('Train audio files got got!')
    print('Getting test audio files...')
    data.get_audio_files(data.test_directory, 'X_test', 'y_test')
    print('Test audio files got got!')
    print('Getting holdout audio files...')
    data.get_audio_files(data.holdout_directory, 'X_holdout', 'y_holdout')
    print('Holdout audio files got got!')

if __name__ == '__main__':
    main()
