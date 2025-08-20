import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency





# Function to check for missing values in a DataFrame
def ttest_missing_vs_not(df, missing_col):
    """
    Perform t-test on all numerical columns comparing groups with missing vs non-missing values
    in the specified missing_col.

    Parameters:
    - df: pandas DataFrame
    - missing_col: column name (string) to check missing values on

    Returns:
    - results: list of tuples (column, t_statistic, p_value, significance)
    """

    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if missing_col in num_cols:
        num_cols.remove(missing_col)

    group_missing = df[df[missing_col].isna()]
    group_not_missing = df[df[missing_col].notna()]

    results = []

    for col in num_cols:
        data_missing = group_missing[col].dropna()
        data_not_missing = group_not_missing[col].dropna()

        if len(data_missing) > 1 and len(data_not_missing) > 1:
            t_stat, p_value = ttest_ind(data_missing, data_not_missing, equal_var=False)
            signif = 'Yes' if p_value < 0.05 else 'No'
            results.append((col, t_stat, p_value, signif))
        else:
            results.append((col, None, None, 'Not enough data'))

    return results











# Function to perform Chi-square test between missingness indicator and categorical columns
def chi2_missing_vs_categorical(df, missing_col):
    """
    Perform Chi-square test between missingness indicator of a numeric column 
    and all categorical columns in the dataframe.

    Parameters:
    - df: pandas DataFrame
    - missing_col: str, name of numeric column with missing values

    Returns:
    - results: list of tuples (categorical_column, chi2_stat, p_value, significance)
    """

    missing_indicator = missing_col + '_missing'
    df = df.copy()
    df[missing_indicator] = df[missing_col].isna().astype(int)

    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    results = []

    for col in cat_cols:
        contingency_table = pd.crosstab(df[missing_indicator], df[col])
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        signif = 'Yes' if p < 0.05 else 'No'
        results.append((col, chi2, p, signif))

    return results







# Function to get outliers based on IQR method
def get_outliers(df):
    Q1 = np.percentile(df, 25)
    Q3 = np.percentile(df, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df < lower_bound) | (df > upper_bound)]








# Function to clip outliers using IQR method
def iqr_clip(df, cols, k=1.5):
    """
    Clips outliers using the IQR method for given columns.
    
    Parameters:
    - df: DataFrame
    - cols: list of columns to clip
    - k: multiplier for IQR (default 1.5)
    - verbose: whether to print before/after stats
    """

    df_clipped = df.copy()
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR
        df_clipped[col] = df[col].clip(lower=lower, upper=upper)
    return df_clipped








# Function to clip outliers using Z-score method
def zscore_clip(df, cols, threshold=3):
    """
    Clips outliers using Z-score method for given columns.
    
    Parameters:
    - df: DataFrame
    - cols: list of columns to clip
    - threshold: Z-score cutoff (default 3)
    - verbose: whether to print before/after stats
    """

    df_clipped = df.copy()
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        z_scores = (df[col] - mean) / std

        # Clip the values
        df_clipped[col] = np.where(
            z_scores > threshold, mean + threshold * std,
            np.where(z_scores < -threshold, mean - threshold * std, df[col])
        )
    return df_clipped






# Function to plot categorical feature distributions
def plot_categorical_distribution(df, cat_cols, ncols=3):
    """   
    Plots the distribution of categorical features using count plots.   
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        cat_cols (list): List of categorical column names to plot.
        ncols (int): Number of columns in the subplot grid.
        figsize (tuple): Size of the figure.
        n_unique (int): Maximum number of unique values to plot for each categorical column.
        rotate (int): Rotation angle for x-axis labels.
    """
    nrows = int(np.ceil(len(cat_cols) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        sns.countplot(x=col, data=df, order=df[col].value_counts().index, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
        axes[i].tick_params(axis='x', rotation=45)

    # Remove empty subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()