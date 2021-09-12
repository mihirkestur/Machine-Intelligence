'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import random


'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float
def get_entropy_of_dataset(df):
    # TODO
    """
    entropy(dataset) = summation [for all target attribute values] (-p/N)log2(p/N)
    """
    # Init entropy to 0
    entropy = 0
    # If dataset is not empty continue, else return 0
    if(df.empty != 1):
        # Assuming target column is last column in dataset, obtain list of target column
        target_column = (df[df.columns[-1]].values).tolist()
        # Note the total samples in the dataframe
        total_samples = len(target_column)
        # Obtain list of unique values in target column
        target_values = set(target_column)
        # For each value in target values, compute the ratio
        for target_value in target_values:
            # Note the count of samples having target attribute as target_value
            positive_sample = target_column.count(target_value)
            # If the positive_sample count is greater than 0 then compute the entropy and accumulate it
            if(positive_sample > 0):
                # Accumulation, (-p/N)log2(p/N)
                entropy = entropy + (-positive_sample/total_samples) * np.log2(positive_sample/total_samples)
    # Return the entropy calculated
    return entropy


'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float
def get_avg_info_of_attribute(df, attribute):
    # TODO
    """
    avg_info(attribute) = summation [for all values of this attribute] (p/N * entropy(attribute_value))
    entropy(attribute_value) = summation [for all target attribute values] (-pos_samples/p)log2(pos_samples/p)
    """
    # Init avg_info to 0
    avg_info = 0
    # If the df is not empty continue
    if(df.empty != 1):
        # Note the columns of the df into a list
        df_columns = df.columns.tolist()
        # Note the total samples in the df
        total_samples = len(df)
        # If there are attributes and the attribute passed is in the df then continue
        if((attribute in df_columns) and (len(df_columns) > 1)):
            # Obtain target column name
            target_column = df_columns[-1]
            # Get unique target column values
            target_values = set((df[target_column].values).tolist())
            # Get unique attribute column values
            attribute_values = set((df[attribute].values).tolist())
            # For every value in attribute values calculate the ratio
            for attribute_value in attribute_values:
                # Making a temp df for every attribute
                df_attribute = df[df[attribute] == attribute_value]
                # For each attribute_value, obtain the total samples in the df_attribute
                total_attribute_samples = len(df_attribute)
                # Init entropy for each value to be 0
                entropy_attribute_value = 0
                # If there are rows with that attribute
                if(total_attribute_samples > 0):
                    # For all values in target_values compute entropy
                    for target_value in target_values:
                        # Obtain the positive samples for each value in target_values
                        positive_samples = len(df_attribute[df_attribute[target_column] == target_value])
                        # If there are any positive samples, then calculate
                        if(positive_samples > 0):
                            # summation [for all target attribute values] (-pos_samples/p)log2(pos_samples/p)
                            entropy_attribute_value = entropy_attribute_value + (-positive_samples / total_attribute_samples * np.log2(positive_samples / total_attribute_samples))
                    # summation [for all values of this attribute] (p/N * entropy(attribute_value))
                    avg_info = avg_info + ((total_attribute_samples/total_samples) * entropy_attribute_value)
    # Return the computed avg_info
    return avg_info


'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float
def get_information_gain(df, attribute):
    # TODO
    # Init IG to 0
    information_gain = 0
    # Obtain avg_info for this attribute
    avg_info_attribute = get_avg_info_of_attribute(df, attribute)
    # Obtain the dataset entropy
    dataset_entropy = get_entropy_of_dataset(df)
    # Calculate information gain => IG = E(dataset) - avg_info(Attribute)
    information_gain = dataset_entropy - avg_info_attribute
    # Return the IG obtained
    return information_gain

#input: pandas_dataframe
#output: ({dict},'str')
def get_selected_attribute(df):
    '''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
    # TODO
    if(df.empty != 1):
        # Init a dictionary to hold IG values
        IG_dictionary = dict()
        # Obtain the columns in df
        df_columns = df.columns.tolist()
        # Continue if there are attributes
        if(len(df_columns) > 1):
            # Obtain the list of attributes
            attributes = df_columns[:-1]
            # For each attribute in the list compute the IG
            for attribute in attributes:
                # Obtain IG for this attribute
                IG_attribute = get_information_gain(df, attribute)
                # Insert it into the dictionary
                IG_dictionary[attribute] = IG_attribute
            # Find the key:value pair having the highest value (i.e. IG) and then note the key
            selected_attribute = max(zip(IG_dictionary.values(), IG_dictionary.keys()))[1]
            # Return the tuple of (dictionary, selected_attribute)
            return tuple((IG_dictionary, selected_attribute))
    # Otherwise return tuple with empty dict and no attribute
    return (dict(), '')