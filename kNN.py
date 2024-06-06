import csv
from statistics import median
from collections import Counter
import os
import math
import random
import copy
import math

#Analyzes the data types of each column, handles missing values
#Fills them with the mode for categorical columns and median for numerical columns.
def analyze_and_fill_csv(data):
    # Initialize a dictionary to store data types for each column
    column_data_types = {}
    modes = {}
    medians = {}

    # Iterate over columns
    for col_idx, col_name in enumerate(data[0]):
        column_values = [row[col_idx] for row in data[1:]]  # Skip header row
        cleaned_values = []

        # Clean values in the column
        for value in column_values:
            if isinstance(value, str) and (value == '' or value.lower() == 'n/a'):
                cleaned_values.append(None)  # Replace empty or 'N/A' with None
            else:
                try:
                    cleaned_values.append(float(value))  # Try converting to float
                except ValueError:
                    cleaned_values.append(value)  # Keep string values

        # Determine majority data type for the column
        numerical_count = sum(isinstance(value, float) for value in cleaned_values)
        categorical_count = len(cleaned_values) - numerical_count

        #If it's a numerical data type, we get the median of the column data.
        if numerical_count >= categorical_count:
            column_data_types[col_name] = 'numerical'
            medians[col_name] = median([value for value in cleaned_values if value is not None])
        #If it's a categorical data type, we get the mode of the column data.
        else:
            column_data_types[col_name] = 'categorical'
            value_counts = Counter(cleaned_values)
            modes[col_name] = max(value_counts, key=value_counts.get)

    # Fill in missing values
    for row_idx, row in enumerate(data[1:], start=1):
        for col_idx, value in enumerate(row):
            col_name = data[0][col_idx]
            if isinstance(value, str) and (value == '' or value.lower() == 'n/a'):
                if column_data_types[col_name] == 'categorical':
                    data[row_idx][col_idx] = modes[col_name]
                else:
                    data[row_idx][col_idx] = medians[col_name]

    return data, column_data_types

#Calculates the Euclidean distance between a training row and a validation row, 
#considering both numerical and categorical features.
#Standardizes distances between 0-1, Uses Jaccard distance for all categorical variables
def euclidean_distance(training_row, validation_row, column_data_types, headers, column_min_max):
    distance = 0
    num_cols = len(training_row)
    t_categorical_values = []
    v_categorical_values = []
    
    for col_idx in range(num_cols - 1):
        col_name = headers[col_idx]
        t_value = training_row[col_idx]
        v_value = validation_row[col_idx]

        if column_data_types[col_name] == 'numerical':
            col_min, col_max = column_min_max[col_name]
            max_difference = col_max - col_min

            if max_difference == 0:
                # If all values are the same, set the normalized value to 0
                t_value = v_value = 0
            else:
                # Normalize numerical values to the [0, 1] range
                t_value = (float(t_value) - col_min) / max_difference
                v_value = (float(v_value) - col_min) / max_difference

            distance += (t_value - v_value) ** 2

        elif column_data_types[col_name] == 'categorical':
            t_categorical_values.extend(t_value.split(','))
            v_categorical_values.extend(v_value.split(','))
    
    #Calculate Jaccard similarity coefficient for all categorical data
    t_set = set(t_categorical_values)
    v_set = set(v_categorical_values)
    intersection = t_set.intersection(v_set)
    union = t_set.union(v_set)
    if len(union) == 0:
        jaccard_similarity = 1  # Handle edge case when both sets are empty
    else:
        jaccard_similarity = len(intersection) / len(union)
    
    distance += 1 - jaccard_similarity

    #Square root
    distance = math.sqrt(distance)
    return distance

#Splits the dataset into training and validation sets based on the specified percentage.
def split_dataset(data, train_percentage):
    # Create a copy of the data to avoid modifying the original
    data_copy = copy.deepcopy(data)
    
    # Remove the header row from the data
    header = data_copy.pop(0)
    
    # Shuffle the data randomly
    random.shuffle(data_copy)
    
    # Calculate the number of rows for the training and validation sets
    total_rows = len(data_copy)
    train_rows = int(total_rows * train_percentage / 100)
    
    # Split the data into training and validation sets
    train_set = [header] + data_copy[:train_rows]
    val_set = [header] + data_copy[train_rows:]
    
    return train_set, val_set

#Calculates the minimum and maximum values for each numerical column in the training set.
def min_max(column_data_types, headers, training_set):
    # Get the minimum and maximum values for each column
    column_min_max = {}
    for col_idx, col_name in enumerate(headers):
        # Check if the column is numerical
        if column_data_types[col_name] == 'numerical':
            # Initialize the minimum and maximum values with the first non-empty numerical value
            col_values = [float(row[col_idx]) for row in training_set[1:] if row[col_idx] != '']
            if col_values:
                col_min = col_max = col_values[0]
            else:
                # If all values are empty, set min and max to None
                col_min = col_max = None

            # Update the minimum and maximum values for the current column
            for value in col_values:
                col_min = min(col_min, value)
                col_max = max(col_max, value)

            # Store the minimum and maximum values in the dictionary
            column_min_max[col_name] = (col_min, col_max)
    return column_min_max

#Performs k-Nearest Neighbors classification on the validation set using the training set, 
#considering the specified selection type (weighted or majority vote).
#Returns a list of predicted class labels
def kNN(k, training_set, validation_set, column_data_types, selection_type):
    predictions = []
    headers = training_set[0]
    # Get the minimum and maximum values for each column
    column_min_max = min_max(column_data_types, headers, training_set)

    # Iterate over validation set rows
    for vrow_idx in range(1, len(validation_set)):
        weighted_labels = {}
        distances = []
        validation_row = validation_set[vrow_idx]

        # For each validation row, we calculate the distance of it against all training rows, then store all the distancse
        for trow_idx in range(1, len(training_set)):
            training_row = training_set[trow_idx]
            distance = euclidean_distance(training_row, validation_row, column_data_types, headers, column_min_max)
            distances.append((distance, training_set[trow_idx][-1]))  # Store a tuple with distance and a class label associated with that distance ex: (2.152, Red Flower)

        distances.sort(key=lambda x: x[0])  # Sort the distances in ascending order
        k_nearest_labels = [label for _, label in distances[:k]]  # Get the labels for the k nearest neighbors

        # Predict the class label based on weighted vote
        if selection_type == 'W':
            i = 0
            for distance, class_label in distances[:k]:
                if distance != 0:
                    weight = 1 / (distance ** 2)
                else:
                    weight = float('inf')
                weighted_labels[k_nearest_labels[i]] = weighted_labels.get(k_nearest_labels[i], 0) + weight
                i += 1 
            
            predicted_label = max(weighted_labels, key=weighted_labels.get)
            predictions.append(predicted_label)
        # Predict the class label based on majority vote
        else:
            class_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(class_label)
    #After finishing we have a list of 'predictions' which is an array of class labels, the index of these correspond to the row in our validation data which we will use it against
    return predictions 

#Calculates the error percentage by comparing the predicted labels with the true labels in the validation set.
def calculate_error(predictions, validation_set):
    total_rows = len(validation_set) - 1  # Exclude header row
    incorrect_predictions = 0
    
    #Goes through our validation set row by row and compares it with our predicted value
    for row_idx, row in enumerate(validation_set[1:], start=1):
        true_label = row[-1]
        predicted_label = predictions[row_idx - 1]
        
        if true_label != predicted_label:
            incorrect_predictions += 1
    #Returns percent
    error_percentage = (incorrect_predictions / total_rows) * 100
    return error_percentage

#Performs k-fold cross-validation on the datasetcalculating the average error percentage and individual fold error percentages.
def cross_validation(data, k, folds, selection_type):
    # Remove the header row from the data
    header = data[0]
    
    # Shuffle the data randomly
    data, _ = split_dataset(data, 100)
    data, column_data_types = analyze_and_fill_csv(data)
    data = data[1:]

    fold_size = len(data) // folds
    error_percentages = []

    for i in range(folds):
        # Split the data into training and validation sets for the current fold
        start = i * fold_size
        end = start + fold_size
        
        validation_set = [header]
        validation_set.extend(data[start:end])
        training_set = [header] 
        training_set.extend(data[:start] + data[end:])
        
        # Make predictions using kNN
        predictions = kNN(k, training_set, validation_set, column_data_types, selection_type)
        
        # Calculate the error percentage for the current fold
        error_percentage = calculate_error(predictions, validation_set)
        error_percentages.append(error_percentage)
    
    # Calculate the average error percentage across all folds
    avg_error_percentage = sum(error_percentages) / folds
    
    return avg_error_percentage, error_percentages

####################################
#            USER INPUT            #
####################################
filename = input("Enter the name of your file(Must be a csv):")
percentage = int(input("Enter the percentage of the data you want to use for training: "))
k = int(input("Enter the number of nearest neighbors k you want to use: "))
selection_type = input("Would you like to use Majority rule or Weighted distance?:('M' / 'W'): ")
crossvalidationoption = input("Would you like to do a cross validation as well?:('Y' / 'N'): ")
if crossvalidationoption == 'Y':
    folds = int(input("Enter the number of folds you want to use: "))
####################################
#            Reading File          #
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, filename)
with open(file_path, 'r', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    data = list(reader)
####################################
training_set, validation_set = split_dataset(data, percentage)
training_set, column_data_types = analyze_and_fill_csv(training_set)
validation_set, _ = analyze_and_fill_csv(validation_set)
####################################
predictions = kNN(k, training_set, validation_set, column_data_types, selection_type)
error_percentage = calculate_error(predictions, validation_set)
print(f"Error percentage: {error_percentage:.2f}%")

if crossvalidationoption == 'Y':
    avg_error_percentage, error_percentages = cross_validation(data, k, folds, selection_type)
    print(f"\nFold Results:")
    print(f"Average Error Percent of Folds: {avg_error_percentage:.2f}%")
    for i, error_percentage in enumerate(error_percentages):
        print(f"Error Percent for Fold {i + 1}: {error_percentage:.2f}%")
