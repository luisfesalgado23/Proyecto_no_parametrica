import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from SyntheticDataCopulas.src.non_parametric.synthetic_data_generator import generate_multivariate_data
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.linear_model import HuberRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from scipy.stats import t

def plot_histograms_separados(dataframe):

    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))  

    axs = axs.flatten()

    for i, col in enumerate(dataframe.columns):
        axs[i].hist(dataframe[col], bins=50)  
        axs[i].set_title(col)

    plt.tight_layout()
    plt.grid()
    plt.show()

def corr_multiples(df):
    
    S = df.cov().values                
    S_inv = np.linalg.inv(S)
    
    Vector_Corre_Multiples = 1 - 1/ (np.diag(S) * np.diag(S_inv))
    Vector_Corre_Multiples.round(4)

    df_correlaciones = pd.DataFrame({'Columna': df.columns,'Correlacion_Multiple': Vector_Corre_Multiples})
    
    return df_correlaciones.sort_values(by="Correlacion_Multiple",ascending=False)

def plot_boxplots_separados(dataframe):

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))  

    axs = axs.flatten()

    for i, col in enumerate(dataframe.columns):
        axs[i].boxplot(dataframe[col])  
        axs[i].set_title(col)

    plt.tight_layout()
    plt.grid(True)
    plt.show()

def outliers_euclidea(X, alfa):
    distances = []
    mu = np.mean(X, axis = 0)
    
    for x in X:
        distances.append(euclidean_distance(x, mu))
    
    distances = np.array(distances)
    cutoff = np.percentile(distances,100-alfa*100)
    
    outliers = (distances > cutoff).astype(int)
    return outliers

def outliers_mahal(X, alfa):
    
    distances = []
    mu = np.mean(X, axis = 0)
    
    cov_matrix = np.cov(X, rowvar=False)
    cov_matrix_inv = np.linalg.inv(cov_matrix)
    
    for x in X:
        distances.append(distance.mahalanobis(x, mu, cov_matrix_inv))
    
    distances = np.array(distances)
    cutoff = np.percentile(distances,100-alfa*100)
    
    outliers = (distances > cutoff).astype(int)
    return outliers

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

def redondeo_personalizado(numero):

    if numero % 1 >= 0.5:
        return int(numero) + 1
    else:
        return int(numero)
    
def plot_y_true_vs_y_pred(y_true, y_pred, title="Gráfico y_true vs y_pred", xlabel="y_true", ylabel="y_pred"):

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, edgecolors=(0, 0, 0))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def plot_error_distribution(y_true, y_pred, title="Distribución de Errores", xlabel="Errores", ylabel="Frecuencia"):

    errores = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.hist(errores, bins=30, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def mean_absolute_deviation(y_true, y_pred):

    n = len(y_true)
    mad = np.sum(np.abs(y_true - y_pred)) / n
    return mad

def huber_loss(u, c):
    return np.where(np.abs(u) <= c, 0.5 * u**2, c * (np.abs(u) - 0.5 * c))

def tukey_bisquare_weights(u, k):
    return np.where(np.abs(u) <= k, (1 - (u / k)*2)*2, 0)

def loss_function(beta, X, y, loss_fn, weights, c):
    residuals = y - X.dot(beta)
    return np.sum(weights * loss_fn)

def fit_m_estimator(X, y, c, k, max_iterations=100, tol=1e-4):
    n, p = X.shape
    beta = np.zeros(p)

    result = minimize(lambda b: loss_function(b, X, y, huber_loss(X.dot(b) - y, c), tukey_bisquare_weights(X.dot(b) - y, k), c),
                      beta, method='L-BFGS-B', options={'disp': True, 'maxiter': max_iterations, 'gtol': tol})
    
    return result.x/100

def l1_loss(u):
    return np.sum(np.abs(u))

def l1_regression_loss(beta, X, y):
    residuals = y - X.dot(beta)
    return l1_loss(residuals)

def l1_regression(X, y, max_iterations=100, tol=1e-4):
    n, p = X.shape
    beta = np.zeros(p)

    result = minimize(lambda b: l1_regression_loss(b, X, y), beta, method='L-BFGS-B', options={'disp': True, 'maxiter': max_iterations, 'gtol': tol})
    
    return result.x

def int_conf_no_par(x,y):
    datos_1=pd.concat([pd.DataFrame(x),pd.DataFrame(y)], axis=1)

    B = 500
    k = int(len(datos_1)*0.8)


    param_1 = []

    for i in range(B):
        sample_1 = datos_1.sample(n = k)
        param_1.append(mean_absolute_deviation(sample_1.iloc[:,0],sample_1.iloc[:,1]))

        
    return (np.quantile(param_1,0.01),np.quantile(param_1,0.99))

def int_conf_sem_par(y):

    min = np.min(y)

    # Realiza el procedimiento de Jackknife
    jackknife_mins = []

    for i in range(len(y)):
        jackknife_data = np.delete(y.values, i)
        jackknife_min = np.min(jackknife_data)
        jackknife_mins.append(jackknife_min)

    # Estimación de la varianza
    jackknife_std = np.std(jackknife_mins, ddof=1)


    jackknife_bias = (len(y) - 1) * (np.min(jackknife_mins) - min)

    # Grados de libertad
    df = len(y) - 1

    # Nivel de confianza
    alpha = 0.01

    # Intervalo de confianza
    t_critical = t.ppf(1 - alpha / 2, df)
    lower_limit = min - t_critical * jackknife_std
    upper_limit = min + t_critical * jackknife_std

def add_outliers(data, num_outliers=50, magnitude=10):

    outlier_indices = np.random.choice(1002, num_outliers, replace=False)
    dataset_out = np.copy(data)
    dataset_out[outlier_indices] += magnitude * np.random.randn(num_outliers, 13)
    dataset_out = pd.DataFrame(data=dataset_out, columns=data.columns.values)

    return dataset_out

