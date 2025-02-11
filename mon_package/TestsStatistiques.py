import pandas as pd
from scipy.stats import chi2_contingency
from scipy import stats

def chi2(X, correction=True):
    # Initialiser une liste pour stocker les résultats
    results_list = []
    
    # Obtenir les colonnes du DataFrame
    columns = X.columns
    
    # Boucle imbriquée pour générer toutes les paires de colonnes
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]
            
            # Créer une table de contingence
            contingency = pd.crosstab(X[col1], X[col2])
            
            # Effectuer le test du chi²
            chi2, p_val, dof, exp_freq = chi2_contingency(contingency.values, correction=correction)
            
            # Ajouter les résultats à la liste
            results_list.append([col1, col2, chi2, p_val, dof])
    
    # Convertir les résultats en DataFrame
    results = pd.DataFrame(
        results_list,
        columns=['column1', 'column2', 'chi_squared', 'p_value', 'dof']
    )
    
    return results



def anova(X):
    numeric_columns = X.select_dtypes(include=['number']).columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    results = []
    
    for num_col in numeric_columns:
        for cat_col in categorical_columns:
            groups = [X[X[cat_col] == category][num_col].dropna().values 
                      for category in X[cat_col].unique()]
            
            f_stat, p_val = stats.f_oneway(*groups)
            results.append([num_col, cat_col, f_stat, p_val])
    
    return pd.DataFrame(results, columns=['num_column', 'cat_column', 'f_stat', 'p_value'])


