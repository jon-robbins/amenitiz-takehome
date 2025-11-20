"""
Visualization functions for ML project.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribution(data, column, title="Distribution"):
    """
    Plot distribution of a column.
    
    Args:
        data: DataFrame containing the data
        column: Column name to plot
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    return plt


def plot_correlation_matrix(data, title="Correlation Matrix"):
    """
    Plot correlation matrix.
    
    Args:
        data: DataFrame containing the data
        title: Title for the plot
    """
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(title)
    return plt