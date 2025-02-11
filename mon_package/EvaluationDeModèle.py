import pandas as pd
from numpy import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import  mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# Fonction pour calculer les performances du modèle
def calc_model_performance(y_true, y_pred):
    results = {}
    results['Root Mean Squared Error'] = sqrt(mean_squared_error(y_true, y_pred))
    results['Mean Squared Error'] = mean_squared_error(y_true, y_pred)
    results['Mean Absolute Error'] = mean_absolute_error(y_true, y_pred)
    results['Mean Absolute Percentage Error'] = mean_absolute_percentage_error(y_true, y_pred)
    results['R Squared'] = r2_score(y_true, y_pred)
    return results

# Fonction pour comparer les performances de deux modèles
def compare_model_performance(base_perf, new_perf):
    results = pd.DataFrame(columns=['base', 'new', 'abs_improvement', 'perc_improvement'])
    for metric, base_value in base_perf.items():
        base_value = round(base_value, 2)
        new_value = round(new_perf[metric], 2)
        results.loc[metric] = [
            base_value,
            new_value,
            new_value - base_value,
            round(100 * (new_value - base_value) / base_value, 2)
        ]
    return results

# Fonction pour visualiser l'homoscedasticité des résidus
def compare_homoscedasticity(y_true, y_pred_base, y_pred_new):
    res_base = y_true - y_pred_base
    res_new = y_true - y_pred_new

    # Tracé des résidus
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=y_true, y=res_base, label='Base Model Residuals', alpha=0.7)
    sns.scatterplot(x=y_true, y=res_new, label='New Model Residuals', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.title('Comparison of Residual Homoscedasticity')
    plt.legend()
    plt.show()

# Fonction pour calculer le pourcentage de prédictions dans une plage de résidus absolus
def calc_preds_in_residual_range(y_true, y_pred, range_):
    residuals = abs(y_true - y_pred)
    return 100 * (residuals <= range_).mean()

# Fonction pour calculer le pourcentage de prédictions dans une plage de résidus en pourcentage
def calc_preds_in_residual_perc_range(y_true, y_pred, perc_range):
    perc_residuals = 100 * (abs(y_true - y_pred) / y_true)
    return 100 * (perc_residuals <= perc_range).mean()
