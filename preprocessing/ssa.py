from operator import itemgetter
from random import randint
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz
from numpy import linalg as lin


class SingularSpectrumAnalysis():
    def __init__(self, input_signal, window, verbose=False):
        self.input_signal = input_signal
        self.M = window   # window length = embedding dimension
        self.N = self.input_signal.size   # length of generated time series
        self.T = 22    # period length of sine function
        self.number_of_lags = self.N - self.M + 1
        self.std_noise = 1 # noise-to-signal ratio
        self.verbose =verbose
        self.covariance_matrix = []
        self.trajectory_matrix = []
        self.eigne_vectors = []
        self.eigne_values = []
        self.principal_components = []
        self.reconstructed_matrix = []

    def nomalize(self):
        mean = np.mean(self.input_signal, axis=0)
        self.input_signal = self.input_signal - mean
        self.input_signal = self.input_signal / np.std(self.input_signal, axis=0)
        if self.verbose:
            plt.figure(1)
            plt.subplot(311)
            plt.plot(self.input_signal)
            plt.title('Normalized Values')
            plt.grid(True)
    def calculate_trajectory_matrix(self):
        self.trajectory_matrix = np.zeros([self.N - self.M + 1, self.M])
        for m in range(0, self.N - self.M + 1):
            self.trajectory_matrix[m] = self.input_signal[m:self.M + m]

    def calculate_covariance_matrix_toeplitz_approach(self):
        # correlation = plt.xcorr(self.input_signal, self.input_signal, maxlags=self.M - 1)
        correlation = self.coss_corelation(self.input_signal, self.input_signal, self.M - 1)
        self.covariance_matrix = toeplitz(correlation[self.M - 1:2 * self.M - 1])
        if self.verbose:
            plt.subplot(312)
            plt.plot(self.covariance_matrix)
            plt.title('Covariance Matrix: Toeplitz Approach')
            plt.grid(True)

    def calculate_covariance_matrix_trajectory_approach(self):
        self.covariance_matrix = np.matmul(self.trajectory_matrix.transpose(), self.trajectory_matrix) / (self.N - self.M + 1)
        if self.verbose:
            plt.subplot(313)
            plt.plot(self.covariance_matrix)
            plt.title('Trajectory Matrix')
            plt.grid(True)

    def calculate_eigen_vectors_and_values(self):
        [self.eigne_values, self.eigne_vectors] = lin.eig(self.covariance_matrix)
        indices, sorted_value = zip(*sorted(enumerate(self.eigne_values), key=itemgetter(1), reverse=True))
        self.eigne_values = list(sorted_value)  # % sort eigenvalues
        self.eigne_vectors = np.array(self.eigne_vectors[list(indices)])  # and eigenvectors
        if self.verbose:
            plt.subplot(321)
            plt.plot(np.real(self.eigne_values))
            plt.title('Eigen Values')
            plt.grid(True)

            plt.subplot(322)
            plt.plot(np.real(self.eigne_vectors))
            plt.title('Eigen Vectors')
            plt.grid(True)

    def calculate_principle_components(self):
        self.principal_components = np.matmul(self.trajectory_matrix, self.eigne_vectors)
        if self.verbose:
            plt.subplot(323)
            plt.plot(self.principal_components)
            plt.title('Principle Components')
            plt.grid(True)

    def reconstruct_matrix(self):
        self.reconstructed_matrix = np.zeros([self.N, self.M])
        for m in range(0, self.M):
            buf = np.outer(self.principal_components[:, m], self.eigne_vectors[:, m].transpose())
            buf = np.array(list(reversed(buf)))
            for n in range(0, self.N):
                self.reconstructed_matrix[n, m] = np.mean(np.real(np.diag(buf, -(self.N - self.M) + n)))

    def coss_corelation(self, x, y, max_lags):
        Nx = len(x)
        if Nx != len(y):
            raise ValueError('x and y must be equal length')
        x = np.asarray(x)
        y = np.asarray(y)
        c = np.correlate(x, y, mode=2)
        c /= np.sqrt(np.dot(x, x) * np.dot(y, y))
        if max_lags >= Nx or max_lags < 1:
            raise ValueError('maglags must be None or strictly '
                             'positive < %d' % Nx)
        lags = np.arange(-max_lags, max_lags + 1)
        c = c[Nx - 1 - max_lags:Nx + max_lags]
        return c


    def get_reconstructed_signal(self, start=0, end=None):
        if(end == None):
            end = self.N

        reconstructed_final_signal = np.sum(self.reconstructed_matrix[:,start:end], axis=1)
        if self.verbose:
            plt.subplot(331)
            plt.plot(reconstructed_final_signal)
            plt.title('Reconstructed Signal')
            plt.grid(True)
            plt.subplot(332)
            plt.plot(self.input_signal)
            plt.title('Noise Added Signal')
            plt.grid(True)
            plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                                wspace=0.35)
            plt.show()
        return reconstructed_final_signal

    def execute(self):
        self.nomalize()
        self.calculate_trajectory_matrix()
        self.calculate_covariance_matrix_toeplitz_approach()
        self.calculate_eigen_vectors_and_values()
        self.calculate_principle_components()
        self.reconstruct_matrix()
        return self.get_reconstructed_signal()




