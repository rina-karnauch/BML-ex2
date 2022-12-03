import numpy as np
from matplotlib import pyplot as plt
from typing import Callable

SIGMA = 0.25


def polynomial_basis_functions(degree: int) -> Callable:
    """
    Create a function that calculates the polynomial basis functions up to (and including) a degree
    :param degree: the maximal degree of the polynomial basis functions
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             polynomial basis functions, a numpy array of shape [N, degree+1]
    """

    def pbf(x: np.ndarray):
        design_matrix = np.empty(shape=(len(x), degree + 1))
        for p_i in range(degree + 1):
            for i, x_i in enumerate(x):
                design_matrix[i, p_i] = pow(x_i, p_i)
        return design_matrix

    return pbf


def gaussian_basis_functions(centers: np.ndarray, beta: float) -> Callable:
    """
    Create a function that calculates Gaussian basis functions around a set of centers
    :param centers: an array of centers used by the basis functions
    :param beta: a float depicting the lengthscale of the Gaussians
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             Gaussian basis functions, a numpy array of shape [N, len(centers)+1]
    """

    def gbf(x: np.ndarray):
        new_centers = np.array([0] + list(centers))
        design_matrix = np.empty(shape=(len(x), len(new_centers)), dtype=np.float32)
        for c_i, c in enumerate(new_centers):
            for x_i, x_v in enumerate(x):
                exp_val = (((x_v - c) ** 2) / (2 * (beta ** 2))) if c_i > 0 else 0
                design_matrix[x_i, c_i] = np.exp(-exp_val)
        return design_matrix

    return gbf


def spline_basis_functions(knots: np.ndarray) -> Callable:
    """
    Create a function that calculates the cubic regression spline basis functions around a set of knots
    :param knots: an array of knots that should be used by the spline
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             cubic regression spline basis functions, a numpy array of shape [N, len(knots)+4]
    """

    def csbf(x: np.ndarray):
        design_matrix = np.empty(shape=(len(x), len(knots) + 4))
        for i in range(4):
            for x_i, x_v in enumerate(x):
                design_matrix[x_i, i] = pow(x_v, i)
        for c_i, c in enumerate(knots):
            knot_f = lambda t: 0 if (t - c) < 0 else pow(t - c, 3)
            for x_i, x_v in enumerate(x):
                knot_f_val = knot_f(x_v)
                design_matrix[x_i, c_i + 4] = knot_f_val
        return design_matrix

    return csbf


def learn_prior(hours: np.ndarray, temps: np.ndarray, basis_func: Callable) -> tuple:
    """
    Learn a Gaussian prior using historic data
    :param hours: an array of vectors to be used as the 'X' data
    :param temps: a matrix of average daily temperatures in November, as loaded from 'jerus_daytemps.npy', with shape
                  [# years, # hours]
    :param basis_func: a function that returns the design matrix of the basis functions to be used
    :return: the mean and covariance of the learned covariance - the mean is an array with length dim while the
             covariance is a matrix with shape [dim, dim], where dim is the number of basis functions used
    """
    thetas = []
    # iterate over all past years
    for i, t in enumerate(temps):
        ln = LinearRegression(basis_func).fit(hours, t)
        theta_i = ln.get_theta()
        thetas.append(theta_i)  # append learned parameters here

    thetas = np.array(thetas)

    # take mean over parameters learned each year for the mean of the prior
    mu = np.mean(thetas, axis=0)
    # calculate empirical covariance over parameters learned each year for the covariance of the prior
    cov = (thetas - mu[None, :]).T @ (thetas - mu[None, :]) / thetas.shape[0]
    return mu, cov


class BayesianLinearRegression:
    def __init__(self, theta_mean: np.ndarray, theta_cov: np.ndarray, sig: float, basis_functions: Callable):
        """
        Initializes a Bayesian linear regression model
        :param theta_mean:          the mean of the prior
        :param theta_cov:           the covariance of the prior
        :param sig:                 the signal noise to use when fitting the model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self._theta_mean = theta_mean
        self._theta_cov = theta_cov
        self._sig = sig
        self._bf = basis_functions
        self._H = None
        self._posterior_mean = None
        self._posterior_cov = None
        self._inv_cov_mu = None
        self._sigma_HT_y = None
        self._C_theta_D = None
        self._theta_MMSE = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        self._H = self._bf(X)
        _HT = np.transpose(self._H)
        _inv_cov = np.linalg.inv(self._theta_cov)
        self._inv_cov_mu = _inv_cov @ self._theta_mean
        self._sigma_HT_y = (1 / self._sig) * (_HT @ y)
        _I = np.identity(len(self._theta_cov))
        self._C_theta_D = np.linalg.solve(_inv_cov + (1 / self._sig) * _HT @ self._H, _I)
        self._theta_MMSE = self._C_theta_D @ (self._sigma_HT_y + self._inv_cov_mu)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model using MMSE
        :param X: the samples to predict
        :return: the predictions for X
        """
        H_x = self._bf(X)
        y_predicted = H_x @ self._theta_MMSE
        return y_predicted

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        H_x = self._bf(X)
        std = np.array([np.sqrt((h_x.T @ self._C_theta_D @ h_x) + self._sig) for h_x in H_x])
        return std

    def posterior_sample(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model and sampling from the posterior
        :param X: the samples to predict
        :return: the predictions for X
        """
        # mu = _theta_MMSE, cov = C_theta_D
        random_f = np.random.multivariate_normal(self._theta_MMSE, self._C_theta_D, 5)
        H_x = self._bf(X)
        prediction = np.array([H_x @ r_f for r_f in random_f])
        return prediction

    def get_MMSE(self):
        return self._theta_MMSE


class LinearRegression:

    def __init__(self, basis_functions: Callable):
        """
        Initializes a linear regression model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self._bf = basis_functions
        self._H = None
        self._y = None
        self._X = None
        self._H_dagger = None
        self._noise = None
        self._thetaMLE = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the model to the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        self._X = X
        self._y = y
        self._noise = gaussian_noise_getter(sigma=SIGMA, n=len(self._y))
        self._H = self._bf(self._X)
        self._H_dagger = np.linalg.pinv(self._H)
        self._thetaMLE = self._H_dagger @ self._y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        H_x = self._bf(X)
        y_predicted = H_x @ self._thetaMLE \
            # + self._noise
        return y_predicted

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the model and return the predicted values for X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def get_theta(self):
        return self._thetaMLE


def gaussian_noise_getter(sigma: float, n: int):
    return np.random.multivariate_normal(np.zeros(shape=n), np.diag([sigma] * n))


def plot_prior(mu, cov, f, p):
    x = np.arange(0, 24, 0.1)
    H_x = f(x)
    y = H_x @ mu
    ci = [np.sqrt(h_x.T @ cov @ h_x + SIGMA) for h_x in H_x]
    plot_dist(x, y, ci, H_x, mu, cov, p)


def plot_dist(x, y, ci, H_x, mu, cov, p):
    plot_colors = ["mediumpurple", "purple", "rebeccapurple", "indigo", "darkorchid"]
    plt.figure()
    plt.fill_between(x, y - ci, y + ci, color="thistle", alpha=.5, label='confidence interval')

    random_f = np.random.multivariate_normal(mu, cov, 5)
    for i, f in enumerate(random_f):
        prediction_f = H_x @ f
        plt.plot(x, prediction_f, label=('random function #' + str(i)), color=plot_colors[i])
    plt.plot(x, y, 'k', lw=2, label='prior mean')
    plt.legend()
    plt.xlabel(r'$hour$')
    plt.ylabel(r'$temperature$')
    plt.title(f"bayesian linear regression, prior distribution, {p}")
    plt.show()


def plot_bayesian_model(model, train_hours, train, test_hours, test, parameter, label):
    # train model
    model.fit(train_hours, train)
    y_predicted = model.predict(test_hours)
    train_predicted = model.predict(train_hours)

    ci = model.predict_std(test)
    ci_t = model.predict_std(train)

    posterior_samples = model.posterior_sample(test_hours)
    posterior_samples_t = model.posterior_sample(train_hours)

    # print average squared error performance
    SE = np.mean((test - y_predicted)**2)
    print(f'Average squared error with BLR and {parameter} is {SE:.2f}')

    # plot everything
    plot_colors = ["mediumseagreen", "forestgreen", "olivedrab", "seagreen", "green"]
    plt.figure()
    plt.title(label + f" :bayesian linear regression, prior distribution, {parameter}" + " E:" + "{:.2f}".format(SE))
    plt.fill_between(test_hours, y_predicted - ci, y_predicted + ci, color="palegreen", alpha=.5,
                     label='confidence interval')
    plt.fill_between(train_hours, train_predicted - ci_t, train_predicted + ci_t, color="palegreen", alpha=.5,
                     label='confidence interval')
    for i, p in enumerate(posterior_samples):
        plt.plot(test_hours, p, color=plot_colors[i])
    for i, p_t in enumerate(posterior_samples_t):
        plt.plot(train_hours, p_t, color=plot_colors[i])
    plt.plot(test_hours, y_predicted, 'k', lw=2, label='MMSE prediction')
    plt.scatter(test_hours, test, label='test data', color="darkolivegreen")
    plt.scatter(train_hours, train, label='train data', color="yellowgreen")
    plt.legend()
    plt.xlabel(r'$hour$')
    plt.ylabel(r'$temperature$')
    plt.xlim([0, 24])

    plt.show()


def train_and_plot_linear_regression(f: Callable, d: int, train_hours, train, test_hours, test, label):
    ln = LinearRegression(f(d)).fit(train_hours, train)
    y_predicted = ln.predict(test_hours)

    # print average squared error performance
    SE = np.mean((test - y_predicted) ** 2)
    print(f'Average squared error with LR and d={d} is {SE:.2f}')

    # plot graphs for linear regression part
    plt.plot(test_hours, y_predicted, color="slategrey", label="prediction")
    plt.scatter(test_hours, test, color="powderblue", label="test set")
    plt.scatter(train_hours, train, color="slategray", label="train set")
    plt.legend()
    plt.title(label + " linear regression temperature predictions, deg:" + str(d) + " E:" + "{:.2f}".format(SE))
    plt.xlabel(r'$hour$')
    plt.ylabel(r'$temperature$')
    plt.show()


def main():
    # load the data for November 16 2020
    nov16 = np.load('nov162020.npy')
    nov16_hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16) // 2]
    train_hours = nov16_hours[:len(nov16) // 2]
    test = nov16[len(nov16) // 2:]
    test_hours = nov16_hours[len(nov16) // 2:]

    # setup the model parameters
    degrees = [3, 7]

    # ----------------------------------------- Classical Linear Regression
    for d in degrees:
        train_and_plot_linear_regression(polynomial_basis_functions, d, train_hours, train, test_hours, test,
                                         "polynomial")

    # ----------------------------------------- Bayesian Linear Regression

    # load the historic data
    temps = np.load('jerus_daytemps.npy').astype(np.float64)
    hours = np.array([2, 5, 8, 11, 14, 17, 20, 23]).astype(np.float64)
    x = np.arange(0, 24, .1)

    # setup the model parameters
    sigma = 0.25
    degrees = [3, 7]  # polynomial basis functions degrees
    beta = 2.5  # lengthscale for Gaussian basis functions

    # sets of centers S_1, S_2, and S_3
    centers = [np.array([6, 12, 18]),
               np.array([4, 8, 12, 16, 20]),
               np.array([2, 4, 8, 12, 16, 20, 22])]

    # sets of knots K_1, K_2 and K_3 for the regression splines
    knots = [np.array([12]),
             np.array([8, 16]),
             np.array([6, 12, 18])]

    # ---------------------- polynomial basis functions

    for deg in degrees:
        pbf = polynomial_basis_functions(deg)
        mu, cov = learn_prior(hours, temps, pbf)

        # plot prior graphs
        plot_prior(mu, cov, pbf, f'deg={deg}')

        # plot posterior graphs
        blr = BayesianLinearRegression(mu, cov, sigma, pbf)

        plot_bayesian_model(blr, train_hours, train, test_hours, test, f'deg={deg} ', "polynomials")

    # ---------------------- Gaussian basis functions

    for ind, c in enumerate(centers):
        rbf = gaussian_basis_functions(c, beta)
        mu, cov = learn_prior(hours, temps, rbf)

        blr = BayesianLinearRegression(mu, cov, sigma, rbf)

        # plot prior graphs
        plot_prior(mu, cov, rbf, f'S{ind + 1}')

        # plot posterior graphs
        plot_bayesian_model(blr, train_hours, train, test_hours, test, f'S{ind + 1} ', "gaussians")

    # ---------------------- cubic regression splines

    for ind, k in enumerate(knots):
        spline = spline_basis_functions(k)
        mu, cov = learn_prior(hours, temps, spline)

        blr = BayesianLinearRegression(mu, cov, sigma, spline)

        # plot prior graphs
        plot_prior(mu, cov, spline, f'K{ind + 1}')

        # plot posterior graphs
        plot_bayesian_model(blr, train_hours, train, test_hours, test, f'K{ind + 1} ', "Spline")


if __name__ == '__main__':
    main()
    print("done")
