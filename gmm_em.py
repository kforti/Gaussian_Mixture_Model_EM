import pandas as pd
import numpy as np

from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

############################################
### Setup
############################################

def readData(filename, cluster=False):
    df = pd.read_csv(filename, header=None)
    clusters = None
    if cluster:
        clusters = df.pop(2)
    points = np.array(df)
    return points, clusters


class GaussianMMEM:
    def __init__(self, points, num_clusters, iterations, epsilon=None):
        self.points = points
        self.mu = None
        self.pi = None
        self.sigma = None
        self.epsilon = epsilon
        self.num_clusters = num_clusters
        self.iterations = iterations

        # log-likelihood per iteration
        self.log_likelihoods = []

    def run(self, display_initial=True):
        # Helps prevent Gaussian collapse
        self.reg_cov = 1e-6 * np.identity(len(self.points[0]))

        x_grid, y_grid = np.meshgrid(np.sort(self.points[:, 0]), np.sort(self.points[:, 1]))
        self.points_XY = np.array([x_grid.flatten(), y_grid.flatten()]).T

        self.initialize_gaussians(display_initial)

        for i in range(self.iterations):
            probabilities, likelihood = self.expectation()

            self.maximization(probabilities)
            #
            if i > 0:
                self.log_likelihoods.append(likelihood)

        probabilities, likelihood = self.expectation()
        self.log_likelihoods.append(likelihood)
        self.display_log_likelihoods()
        self.display_state("Final State")

    def display_log_likelihoods(self):
        fig = plt.figure(figsize=(10, 10))
        subp = fig.add_subplot(111)
        subp.set_title('Log-likelihood')
        subp.plot(range(0, len(self.log_likelihoods), 1), self.log_likelihoods)
        fig.show()

    def calc_log_likelihood(self):
        gaussians = []
        for i, pi in enumerate(self.pi):
            gaussians.append(multivariate_normal(self.mu[i], self.sigma[i]))
        tot = 0
        for p in self.points:
            ptot = 0
            for pi, gauss in zip(self.pi, gaussians):
                ptot += pi * gauss.pdf(p)
            tot = np.log(ptot)
        return tot

    def expectation(self):
        r_ic = np.zeros((len(self.points), self.num_clusters))

        # Compute probabilities for each point and total probabilities
        total_probs = np.zeros(len(self.points), dtype='float')
        for mu, cov, pi, c in zip(self.mu, self.sigma, self.pi, range(self.num_clusters)):
            cov += self.reg_cov
            multi_norm = multivariate_normal(mu, cov)
            probs = multi_norm.pdf(self.points) * pi
            total_probs += probs
            r_ic[:, c] = probs

        # log likelihood
        cprobs = np.sum(r_ic, axis=1)
        p = np.log(cprobs[:])
        likelihood = np.sum(p)

        # Normalize
        for c in range(self.num_clusters):
            r_ic[:, c] = r_ic[:, c] / total_probs

        return r_ic, likelihood

    def maximization(self, r_ic):
        # compute gaussian parameters
        self.mu = []
        self.sigma = []
        self.pi = []

        for c in range(self.num_clusters):
            # Can think of as the fraction of points allocated to cluster
            m_c = np.sum(r_ic[:, c], axis=0)
            mu_c = (1/m_c) * np.sum(self.points * r_ic[:, c].reshape(len(self.points), 1), axis=0)
            self.mu.append(mu_c)

            cov = ((1/m_c) * np.dot(
                (np.array(r_ic[:, c]).reshape(len(self.points), 1) * (self.points - mu_c)).T, (self.points - mu_c))) + self.reg_cov
            self.sigma.append(cov)

            self.pi.append(m_c / np.sum(r_ic))

    def initialize_gaussians(self, display):
        self.mu = np.random.randint(min(self.points[:, 0]), max(self.points[:, 0]), size=(self.num_clusters, len(self.points[0])))
        self.pi = np.ones(self.num_clusters) / self.num_clusters
        self.sigma = np.ones([self.num_clusters, len(self.points[0]), len(self.points[0])])

        for dim in range(self.num_clusters):
            np.fill_diagonal(self.sigma[dim], 5)

        if display:
            self.display_state("Initial State")

    def display_state(self, title):
        fig = plt.figure(figsize=(15, 15))
        subp = fig.add_subplot(111)
        subp.scatter(self.points[:, 0], self.points[:, 1])
        subp.set_title(title)

        for mu, cov in zip(self.mu, self.sigma):
            cov += self.reg_cov
            multi_norm = multivariate_normal(mu, cov)
            subp.contour(np.sort(self.points[:, 0]), np.sort(self.points[:, 1]),
                         multi_norm.pdf(self.points_XY).reshape(len(self.points), len(self.points)), colors='black', alpha=0.3)
            subp.scatter(mu[0], mu[1], c='grey', zorder=10, s=100)
        fig.show()


if __name__ == '__main__':

    points, cluster_labels = readData("data.csv")
    p = points[:500]
    GMM = GaussianMMEM(points, 3, 100)
    GMM.run()