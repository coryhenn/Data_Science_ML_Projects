import os
import sys
import csv
import pandas as pd
import librosa
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Default values
OUTPUT_FILE = 'train_data.csv'
COL_NAMES = True

##
# Function:  getGenres() - get the names of each genre and its index
# Param: dir
# Return: dirnames, labelIndex


def getGenres(dir):
    for (_, dirnames, _) in os.walk(dir):
        labelIndex = np.where(dirnames)  # getting the index of each element
        return (dirnames, labelIndex)

##
# Functions: extractAudioFeatures() - load audio file and extract feature using Librosa
# Param: filePath, genres
# Return: cols - features and target columns


def extractAudioFeatures(filePath, genres):
    global COL_NAMES
    # Load the audio file using Librosa
    # Audio will be automatically resampled to the given rate (default sr=22050).
    # Audio files are at least 30sec in length
    y, sr = librosa.load(filePath, duration=30)
    print("Length of y:", len(y), "Sample Rate:", sr, "Time:", len(y)/sr)
    # sr: float = 22050,n_fft: int = 2048, hop_length: int = 512,

    # Now lets extracts the features as per the 'librosa.feature'
    headers = 'chroma_stft'
    # Compute a chromagram from a waveform or power spectrogram.
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    print("chroma_stft:", chroma_stft.shape)

    # Compute root-mean-square (RMS) value for each frame, either from the audio samples y or from a spectrogram S.
    rms = librosa.feature.rms(y=y)
    print("RMS:", rms.shape)
    headers += f' rms'

    # Compute the spectral centroid. Returns centroid frequencies
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    print("spectral_centroid:", spec_cent.shape)
    headers += f' spectral_centroid'

    # Compute p'th-order spectral bandwidth. Return frequency bandwidth for each frame.
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    print("spectral_bandwidth:", spec_bw.shape)
    headers += f' spectral_bandwidth'

    # Compute roll-off frequency.
    # The roll-off frequency is defined for each frame as the center frequency for a spectrogram bin such that at least roll_percent (0.85 by default) of the energy of the spectrum in this frame is contained in this bin and the bins below. This can be used to, e.g., approximate the maximum (or minimum) frequency by setting roll_percent to a value close to 1 (or 0).
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    print("spectral_rolloff:", rolloff.shape)
    headers += f' spectral_rolloff'

    # Compute the zero-crossing rate of an audio time series.
    zcr = librosa.feature.zero_crossing_rate(y)
    print("zero_crossing_rate:", zcr.shape)
    headers += f' zero_crossing_rate'

    # Mel-frequency cepstral coefficients (MFCCs). Return MFCC sequence
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_shape = mfccs.shape
    print("mfcc:", mfcc_shape)
    for index in range(mfcc_shape[0]):
        headers += f' mfcc-{index+1}'
    headers += ' target'
    print("Headers:", headers)

    # Append the mean feature values in columnwise
    cols = f'{np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    # For each row of mfccs, create one column for each mean value of mfccs row.
    for e in mfccs:
        cols += f' {np.mean(e)}'
    #print("to_append:", to_append)
    # Append the targe column value as per the genra
    label = os.path.basename(os.path.dirname(filePath))
    print("Label:", label)
    print("Genres:", genres)
    genreIndex = genres.index(label)
    cols += f' {genreIndex}'
    #print("Row Values:", cols)

    # Write header to the datafile.
    if COL_NAMES:
        with open(OUTPUT_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers.split())
            COL_NAMES = False

    return cols

##
# Functions: createCSVdata() - creates csv file
# Param: dir, genres
# Return: none


def createCSVdata(dir, genres):
    for root, _, files in os.walk(dir):
        print("Dir: ", root, ", # of files:", len(files))
        for file in files:
            #print("Files: ", files)
            if file.endswith(".au"):
                # Full path to the audio file
                audio_file_path = os.path.join(root, file)
                print("Processing file:", audio_file_path)

                rowValues = extractAudioFeatures(audio_file_path, genres)

                with open(OUTPUT_FILE, 'a', newline='') as out_file:
                    out_writer = csv.writer(out_file)
                    out_writer.writerow(rowValues.split())

##
# Function:  main() - drives to create sample data file
# Param: none
# Return: none


def main():
    cmdlineargs = len(sys.argv)
    if cmdlineargs < 2:
        print("Usage ", sys.argv[0], "<Training data directory path>")
        print("Example:", sys.argv[0], "./data/train")
        print("Etc. Please try again ....")
        sys.exit()
    # Default value
    train_directory = sys.argv[1]
    if not os.path.exists(train_directory):
        print("{} directory does not exits.".format(train_directory))
        sys.exit()

    print("Training Data directory: ", train_directory)

    # Read the Genres
    genres, genreIndex = getGenres(train_directory)
    print("Geners", genres)
    print("GenreIndex:", genreIndex)

    # Read and create datafile in CSV format for modeling
    createCSVdata(train_directory, genres)


# __name__ is a special variable whose value is '__main__' if the module is being run as a script,
# and the module name if it's imported.
if __name__ == "__main__":
    main()
