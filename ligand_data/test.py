import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest, uniform

def plot_tsv_histogram_and_uniform_deviation_rowwise(file_path):
    """
    Loads a TSV file, plots a histogram for each row, and measures deviation from a uniform distribution.

    Args:
        file_path (str): The path to the TSV file.
    """
    try:
        df = pd.read_csv(file_path, sep='\t', index_col=0)

        for index, row in df.iterrows():
            values = row.values

            # Normalize values to the range [0, 1]
            min_val = np.min(values)
            max_val = np.max(values)
            normalized_values = (values - min_val) / (max_val - min_val)

            # Calculate mean and standard deviation
            mean_val = np.mean(normalized_values)
            std_val = np.std(normalized_values)

            # Perform Kolmogorov-Smirnov test for uniformity
            ks_statistic, ks_p_value = kstest(normalized_values, 'uniform')

            # Plot the histogram
            plt.hist(normalized_values, bins='auto', density=True, alpha=0.7, label="Data")
            plt.title(f"Histogram of Normalized Values for Row: {index}")
            plt.xlabel("Normalized Value (0 to 1)")
            plt.ylabel("Frequency (Density)")

            # Plot the uniform distribution for comparison.
            x = np.linspace(0, 1, 100)
            plt.plot(x, uniform.pdf(x), 'r-', label="Uniform Distribution")
            plt.legend()
            plt.show()

            # Print deviation measures
            print(f"Row: {index}")
            print(f"Mean of normalized values: {mean_val:.4f}")
            print(f"Standard deviation of normalized values: {std_val:.4f}")
            print(f"Kolmogorov-Smirnov statistic: {ks_statistic:.4f}")
            print(f"Kolmogorov-Smirnov p-value: {ks_p_value:.4f}")

            # Uniform Distribution Mean and Std
            uniform_mean = 0.5
            uniform_std = np.sqrt(1 / 12)  # standard deviation of a uniform distribution from 0 to 1.

            # Calculate and print differences
            mean_difference = mean_val - uniform_mean
            std_difference = std_val - uniform_std
            print(f"Difference from uniform mean: {mean_difference:.4f}")
            print(f"Difference from uniform std: {std_difference:.4f}")
            print("-" * 40)  # Separator for each row

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
file_path = "LEMBAS/ligand_data/ligandScreen-TFs.tsv"  # Replace with the actual path to your TSV file!

plot_tsv_histogram_and_uniform_deviation_rowwise(file_path)