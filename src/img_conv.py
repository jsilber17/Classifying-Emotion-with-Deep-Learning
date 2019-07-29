import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import glob
import os
import pickle
import cv2


def create_spectrogram(filename,name):
    ''' Create a spectogram of an audio file and save the image to a directory '''

    # Load the audio file using Librosa
    clip, sample_rate = librosa.load(filename, sr=None)

    # Define the paramters for the matplotlib image
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    # Create the mel spectogram and display it using librosa
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

    # Save the figure and delete the matplotlib elements for the next image
    filename  = name + '.jpg'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S


def download_images(glob_path):
    ''' Iterate through a directory that contains audio files and create spectograms for all audio files '''

    directory = np.array(glob.glob(glob_path))
    for actor in directory:
        for file in os.listdir(actor):
            full = actor + '/' + file
            filename, name = full, full.split('/')[-1].split('.')[0]
            create_spectrogram(filename, name)


def get_y_labels(glob_path):
    ''' Get the labels for the audio files which are at position 31 in the audio filename '''

    directory = np.array(glob.glob(glob_path))
    y_train_labels = np.array([img[31] for img in directory])
    return y_train_labels


def dump_pickle(filename, arr):
    ''' Dump a pickle to a directory '''

    with open(filename, 'wb') as f:
        pickle.dump(arr, f)


def main():
    download_images('../data/audio_test/*')
    download_images('../data/audio_train/*')
    y_train = get_y_labels('../data/img_train/*')
    y_test = get_y_labels(',,/data/img_test/*')
    dump_pickle('y_train', y_train)
    dump_pickle('y_test', y_test)

if __name__ == '__main__':
    main()
