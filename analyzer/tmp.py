# Visualize an STFT power spectrum
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load(librosa.util.example_audio_file())
plt.figure(figsize=(12, 8))
number_of_columns = 5
number_of_rows = 2

D = librosa.stft(y)
plt.subplot(number_of_columns, number_of_rows, 1)
librosa.display.specshow(D, y_axis='log')
plt.colorbar()
plt.title('Log-frequency power spectrogram')

# Or on a logarithmic scale
D = np.array(librosa.zero_crossings(y))
D = np.where(D == True, 10, D)
D = np.where(D == False, -10, D)
F=[]
F.append(D.tolist())

plt.subplot(number_of_columns, number_of_rows, 2)
librosa.display.specshow(np.array(F))
plt.colorbar()
plt.title('Zero-Crossing-Rate')

# Or use a CQT scale

CQT = librosa.cqt(y, sr=sr)
plt.subplot(number_of_columns, number_of_rows, 3)
librosa.display.specshow(CQT, y_axis='cqt_note')
plt.colorbar()
plt.title('Constant-Q power spectrogram (note)')

plt.subplot(number_of_columns, number_of_rows, 4)
librosa.display.specshow(CQT, y_axis='cqt_hz')
plt.colorbar()
plt.title('Constant-Q power spectrogram (Hz)')

# Draw a chromagram with pitch classes

tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
plt.subplot(number_of_columns, number_of_rows, 5)
librosa.display.specshow(tonnetz, y_axis='tonnetz')
plt.colorbar()
plt.title('Tonal Centroids (Tonnetz)')

rms = librosa.feature.rmse(y=y)
plt.subplot(number_of_columns, number_of_rows, 6)
plt.semilogy(rms.T, label='RMS Energy')
plt.colorbar()
plt.title('Root Mean Square Energy')

# Draw time markers automatically
cent = librosa.feature.spectral_centroid(y=y, sr=sr)
plt.subplot(number_of_columns, number_of_rows, 7)
plt.semilogy(cent.T, label='Spectral centroid')
plt.ylabel('Hz')
plt.xticks([])
plt.xlim([0, cent.shape[-1]])
plt.colorbar()
plt.title('Spectral centroid')

# Draw a tempogram with BPM markers
plt.subplot(number_of_columns, number_of_rows, 8)
Tgram = librosa.feature.tempogram(y=y, sr=sr)
librosa.display.specshow(Tgram, x_axis='time', y_axis='tempo')
plt.colorbar()
plt.title('Tempogram')
plt.tight_layout()


plt.show()

print ""

