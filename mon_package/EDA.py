import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_histograms(X, height=1200):
    num_features = X.shape[1]
    fig, axes = plt.subplots(num_features, 1, figsize=(10, height / 100))  # Ajuster la hauteur
    if num_features == 1:
        axes = [axes]  # S'assurer que axes est itérable si une seule caractéristique
    for i, col in enumerate(X.columns):
        sns.histplot(data=X, x=col, kde=False, ax=axes[i], color='blue', bins=30)  # Histogramme avec Seaborn
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].grid(True, linestyle='--', alpha=0.6)  # Ajouter une grille
    plt.tight_layout()
    plt.show()
    

def plot_univariate_numeric(X, y):
    num_features = X.shape[1]
    fig, axes = plt.subplots(num_features, 1, figsize=(10, 6 * num_features))  # Ajuster la taille des sous-graphiques
    if num_features == 1:
        axes = [axes]  # S'assurer que `axes` est une liste si une seule caractéristique
    for i, col in enumerate(X.columns):
        sns.scatterplot(x=X[col], y=y, ax=axes[i], color='blue', alpha=0.6)  # Graphique de dispersion
        axes[i].set_title(f"Scatter plot of {col} vs Target")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("charges")
        axes[i].grid(True, linestyle='--', alpha=0.6)  # Ajouter une grille
    plt.tight_layout()
    plt.show()


def plot_univariate_categorical(X, y):
    num_features = X.shape[1]
    fig, axes = plt.subplots(num_features, 1, figsize=(10, 6 * num_features))  # Ajuster la taille des sous-graphiques
    if num_features == 1:
        axes = [axes]  # S'assurer que `axes` est une liste si une seule caractéristique
    for i, col in enumerate(X.columns):
        sns.boxplot(x=X[col], y=y, ax=axes[i], palette='Set2')  # Boxplot avec Seaborn
        axes[i].set_title(f"Boxplot of {col} vs Target")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("charges")
        axes[i].grid(True, linestyle='--', alpha=0.6)  # Ajouter une grille
    plt.tight_layout()
    plt.show()

def plot_heatmap(X, y, bins=10):
    data = pd.concat([X, y], axis=1)
    columns = X.columns
    num_columns = len(columns)
    
    # Discrétisation des colonnes numériques
    for col in X.select_dtypes(include=np.number):
        if X[col].nunique() >= bins:
            data[col] = pd.cut(data[col], bins=bins)

    # Parcourir toutes les paires de colonnes
    for i in range(num_columns):
        for j in range(i + 1, num_columns):  # Éviter les répétitions et les paires identiques
            col1 = columns[i]
            col2 = columns[j]
            
            # Calculer la moyenne de y pour chaque combinaison de col1 et col2
            col_pair_y_mean = data.groupby([col1, col2])[y.name].mean().reset_index()
            col_pair_y_mean = col_pair_y_mean.pivot(index=col1, columns=col2, values=y.name)
            col_pair_y_mean.sort_index(ascending=False, inplace=True)
            
            # Tracer la carte de chaleur avec Seaborn
            plt.figure(figsize=(10, 8))
            sns.heatmap(col_pair_y_mean, cmap='coolwarm', annot=True, fmt='.2f', cbar_kws={'label': 'Mean of Target'}, linewidths=0.5)
            
            # Ajouter des titres et labels
            plt.title(f"Heatmap of {col1} vs {col2}")
            plt.xlabel(col2)
            plt.ylabel(col1)
            
            # Afficher la figure
            plt.tight_layout()
            plt.show()

def plot_paired_boxplots(X, y):
    columns = X.columns
    num_columns = len(columns)
    col_pairs = []
    
    # Générer les paires de colonnes manuellement
    for i in range(num_columns):
        for j in range(i + 1, num_columns):
            col_pairs.append((columns[i], columns[j]))
    
    num_pairs = len(col_pairs)
    fig, axes = plt.subplots(num_pairs, 1, figsize=(12, 6 * num_pairs))  # Ajuster la taille des sous-graphiques
    if num_pairs == 1:
        axes = [axes]  # S'assurer que `axes` est une liste si une seule paire

    for i, (col1, col2) in enumerate(col_pairs):
        # Créer une variable catégorique combinée
        paired_cat = X[col1].astype(str) + ', ' + X[col2].astype(str)
        
        sns.boxplot(x=paired_cat, y=y, ax=axes[i], palette="Set2")  # Boxplot avec Seaborn
        axes[i].set_title(f"Boxplot of {col1} & {col2} vs Target", fontsize=14)
        axes[i].set_xlabel(f"{col1} & {col2}")
        axes[i].set_ylabel("charges")
        axes[i].tick_params(axis='x', rotation=45)  # Rotation des labels pour une meilleure lisibilité
        axes[i].grid(True, linestyle='--', alpha=0.6)  # Ajouter une grille

    plt.tight_layout()
    plt.show()

def plot_paired_scatterplots(X, y):
    data = pd.concat([X, y], axis=1)  # Combiner les données X et y

    num_cols = X.select_dtypes(include=np.number).columns  # Colonnes numériques
    cat_cols = X.select_dtypes(include='object').columns  # Colonnes catégoriques
    
    col_pairs = []
    
    # Générer manuellement les paires (numérique, catégorique)
    for num_col in num_cols:
        for cat_col in cat_cols:
            col_pairs.append((num_col, cat_col))
    
    num_pairs = len(col_pairs)
    fig, axes = plt.subplots(num_pairs, 1, figsize=(12, 6 * num_pairs))  # Taille des sous-graphiques
    if num_pairs == 1:
        axes = [axes]  # S'assurer que `axes` est une liste si une seule paire

    for i, (num_col, cat_col) in enumerate(col_pairs):
        ax = axes[i]
        for cat_val in X[cat_col].unique():
            mask = X[cat_col] == cat_val  # Filtrer par catégorie
            X_filtered = X[mask]
            y_filtered = y[mask]
            
            # Scatterplot pour chaque valeur catégorique
            ax.scatter(
                X_filtered[num_col], y_filtered,
                label=f"{cat_col}={cat_val}", alpha=0.7
            )
        
        # Configurations du graphique
        ax.set_title(f"Scatterplot of {num_col} vs {y.name} grouped by {cat_col}", fontsize=14)
        ax.set_xlabel(num_col)
        ax.set_ylabel(y.name)
        ax.legend(title=cat_col, bbox_to_anchor=(1.05, 1), loc='upper left')  # Légende en dehors du graphique
        ax.grid(True, linestyle='--', alpha=0.6)  # Ajouter une grille

    plt.tight_layout()
    plt.show()



def plot_residuals(y_true, y_pred):
    # Calcul des résidus
    residuals = y_true - y_pred
    
    # Création du graphique
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30, color='skyblue', edgecolor='black')
    
    # Ajout des labels et titre
    plt.title('Distribution of Residuals', fontsize=16)
    plt.xlabel('Residuals', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(visible=True, linestyle='--', alpha=0.6)  # Ajouter une grille pour améliorer la lisibilité
    
    # Afficher le graphique
    plt.tight_layout()
    plt.show()
    
def plot_pearson_wrt_target(X, y):
    # Combiner X et y dans un seul DataFrame
    data = pd.concat([X, y], axis=1)
    
    # Calculer la matrice de corrélation pour les colonnes numériques
    data_corr = data.select_dtypes(include=np.number).corr()
    
    # Extraire les corrélations par rapport à la variable cible (y)
    correlations = data_corr[y.name].drop(y.name)
    
    # Tracer un barplot pour visualiser les corrélations
    plt.figure(figsize=(10, 6))
    sns.barplot(x=correlations.index, y=correlations.values, palette='coolwarm', edgecolor='black')
    
    # Ajouter des titres et labels
    plt.title(f"Pearson Correlation with Target Variable: {y.name}", fontsize=16)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Correlation', fontsize=14)
    plt.xticks(rotation=45, ha='right')  # Rotation des labels pour les rendre lisibles
    
    # Ajouter une ligne horizontale à 0 pour la référence
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    # Afficher le graphique
    plt.tight_layout()
    plt.show()

