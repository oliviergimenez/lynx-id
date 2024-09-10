import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random


def naive_split_dataset(csv_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=None):
    """
    Split a CSV file into train, validation, and test datasets.

    Parameters:
    - csv_path: Path to the CSV file.
    - train_ratio: Proportion of the dataset to include in the train split.
    - val_ratio: Proportion of the dataset to include in the validation split.
    - test_ratio: Proportion of the dataset to include in the test split.
    - random_seed: Optional random seed for reproducibility.

    Returns:
    Paths to the train, validation, and test CSV files.
    """
    from pathlib import Path
    import pandas as pd

    # Verify the sum of ratios is approximately equal to 1
    if not (0.99 <= (train_ratio + val_ratio + test_ratio) <= 1.01):
        return "The sum of train, val, and test ratios must be close to 1."

    dataset_csv_path = Path(csv_path)
    if not dataset_csv_path.is_file():
        return "File does not exist at the specified path."

    # Load the CSV file
    df = pd.read_csv(dataset_csv_path)

    # Shuffle the dataset if random_seed is provided
    if random_seed is not None:
        df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Calculate the number of samples for each split
    total_samples = len(df)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    # Split the dataset
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    # Save the splits to new CSV files
    train_path = dataset_csv_path.parent / f"{dataset_csv_path.stem}_train.csv"
    val_path = dataset_csv_path.parent / f"{dataset_csv_path.stem}_val.csv"
    test_path = dataset_csv_path.parent / f"{dataset_csv_path.stem}_test.csv"

    # train_df.to_csv(train_path, index=False)
    # val_df.to_csv(val_path, index=False)
    # test_df.to_csv(test_path, index=False)

    return train_df, val_df, test_df  # str(train_path), str(val_path), str(test_path)


def plot_occurrence_distribution(df, column_name='lynx_id'):
    """
    Plots the distribution of the number of occurrences of a specified column in a DataFrame.

    Parameters:
    - df: DataFrame containing the dataset.
    - column_name: The name of the column to analyze for occurrence distribution.

    The function will plot a bar chart showing how many times each count of occurrences appears.
    """
    if column_name in df.columns:
        # Count how many times each value in the specified column appears
        value_counts = df[column_name].value_counts()
        # Count how many times each count of occurrence appears
        occurrence_distribution = value_counts.value_counts().sort_index()

        # Plotting
        plt.figure(figsize=(30, 6))
        occurrence_distribution.plot(kind='bar')
        plt.title(f'Distribution of {column_name} Occurrences')
        plt.xlabel('Number of Occurrences')
        plt.ylabel('Frequency')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--')
        plt.show()
    else:
        print(f"The column '{column_name}' does not exist in the dataset.")


def split_and_assign(df_subset, split_ratios, visibility='seen', random_seed=42):
    """
    Splits a subset of the dataframe according to given ratios and assigns visibility and set labels.

    Parameters:
    - df_subset: The DataFrame subset to split.
    - split_ratios: The ratios for splitting the subset into train, val, and test.
    - visibility: Label indicating whether the 'lynx_id' is 'seen' or 'unseen'.
    - random_seed: Seed for random operations to ensure reproducibility.

    Returns:
    Three DataFrame subsets corresponding to train, val, and test splits.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Avoid setting a value on a copy of a slice from a DataFrame.
    df_subset = df_subset.copy()

    # Initialization
    train, val, test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if split_ratios[0] == 0:
        # Handle no train allocation
        if len(df_subset) == 1:
            # Assign single sample to either val or test
            assigned_set = 'val' if random.random() < split_ratios[1] / (split_ratios[1] + split_ratios[2]) else 'test'
            df_subset['set'] = assigned_set
            df_subset['lynx_id_visibility'] = visibility
            if assigned_set == 'val':
                return pd.DataFrame(), df_subset, pd.DataFrame()
            else:
                return pd.DataFrame(), pd.DataFrame(), df_subset
        else:
            val, test = train_test_split(df_subset, train_size=split_ratios[1] / (split_ratios[1] + split_ratios[2]),
                                         random_state=random_seed)
    else:
        if len(df_subset) > 1:
            train, temp = train_test_split(df_subset, train_size=split_ratios[0], random_state=random_seed)
            if len(temp) == 1:
                # If only one sample is left, directly assign it to either val or test based on a random choice.
                temp['set'] = 'val' if random.random() < split_ratios[1] / (
                            split_ratios[1] + split_ratios[2]) else 'test'
                temp['lynx_id_visibility'] = visibility
                if temp['set'].iloc[0] == 'val':
                    val = temp
                else:
                    test = temp
            else:
                val, test = train_test_split(temp, train_size=split_ratios[1] / (split_ratios[1] + split_ratios[2]),
                                             random_state=random_seed)
        else:
            # Single sample for the entire dataset
            assigned_set = np.random.choice(['train', 'val', 'test'], p=split_ratios)
            df_subset['set'] = assigned_set
            df_subset['lynx_id_visibility'] = visibility
            if assigned_set == 'train':
                return df_subset, pd.DataFrame(), pd.DataFrame()
            elif assigned_set == 'val':
                return pd.DataFrame(), df_subset, pd.DataFrame()
            else:
                return pd.DataFrame(), pd.DataFrame(), df_subset

    # Assign set labels and visibility
    train['set'], train['lynx_id_visibility'] = 'train', visibility
    val['set'], val['lynx_id_visibility'] = 'val', visibility
    test['set'], test['lynx_id_visibility'] = 'test', visibility

    return train, val, test


def complex_split_dataset(df, threshold=3, high_occurrence_ratios=(0.7, 0.2, 0.1), low_occurrence_ratios="same",
                          unseen_ratio=0.2, random_seed=42):
    """
    Splits the dataset based on 'lynx_id' occurrence into high and low occurrence groups, then further into
    train, validation, and test sets with special handling for low occurrence 'lynx_id's.

    Parameters:
    - df: DataFrame containing the dataset.
    - threshold: Minimum number of occurrences to be considered high occurrence.
    - high_occurrence_ratios: Tuple of ratios for splitting high occurrence 'lynx_id's into train, val, test.
    - low_occurrence_ratios: Tuple of ratios for splitting seen low occurrence 'lynx_id's into train, val, test,
                              or "same" to use the same as high_occurrence_ratios.
    - unseen_ratio: Ratio for splitting low occurrence 'lynx_id's into seen and unseen.
    - random_seed: Seed for random operations to ensure reproducibility.

    Returns:
    Four DataFrames with an additional 'set' column indicating train, val, or test,
    and 'lynx_id_visibility' indicating seen or unseen.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    if low_occurrence_ratios == "same":
        low_occurrence_ratios = high_occurrence_ratios

    # Calculate 'lynx_id' occurrence
    occurrence_count = df['lynx_id'].value_counts()

    # Determine high and low occurrence 'lynx_id's
    high_occurrence_ids = occurrence_count[occurrence_count >= threshold].index
    low_occurrence_ids = occurrence_count[occurrence_count < threshold].index

    # Initialize empty DataFrames
    train_df, val_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Process high occurrence 'lynx_id's
    for lynx_id in high_occurrence_ids:
        lynx_id_df = df[df['lynx_id'] == lynx_id]
        train, val, test = split_and_assign(lynx_id_df, high_occurrence_ratios)
        train_df = pd.concat([train_df, train])
        val_df = pd.concat([val_df, val])
        test_df = pd.concat([test_df, test])

    # Split low occurrence data into seen and unseen
    seen_low_occurrence_ids, unseen_low_occurrence_ids = train_test_split(low_occurrence_ids,
                                                                          train_size=1 - unseen_ratio,
                                                                          random_state=random_seed)

    # Process unseen low occurrence
    for lynx_id in unseen_low_occurrence_ids:
        lynx_id_df = df[df['lynx_id'] == lynx_id]
        _, val, test = split_and_assign(lynx_id_df, [0, 0.5, 0.5], visibility='unseen')  # Split between val and test
        val_df = pd.concat([val_df, val])
        test_df = pd.concat([test_df, test])

    # Process seen low occurrence
    for lynx_id in seen_low_occurrence_ids:
        lynx_id_df = df[df['lynx_id'] == lynx_id]
        train, val, test = split_and_assign(lynx_id_df, low_occurrence_ratios)
        train_df = pd.concat([train_df, train])
        val_df = pd.concat([val_df, val])
        test_df = pd.concat([test_df, test])

    # Concatenate all dataframes to get a complete dataset
    complete_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

    return train_df, val_df, test_df, complete_df

# Example usage
