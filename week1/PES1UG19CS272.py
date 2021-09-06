#This weeks code focuses on understanding basic functions of pandas and numpy 
#This will help you complete other lab experiments


# Do not change the function definations or the parameters
from types import MethodType
import numpy as np
from numpy.core.defchararray import array
from numpy.core.fromnumeric import shape
import pandas as pd

#input: tuple (x,y)    x,y:int 
def create_numpy_ones_array(shape):
    #return a numpy array with one at all index
    # creates a numpy matrix of dims = shape having all elements as 1
    array = np.ones(shape = shape)
    return array

#input: tuple (x,y)    x,y:int 
def create_numpy_zeros_array(shape):
    #return a numpy array with zeros at all index
    # creates a numpy matrix of dims = shape having all elements as 0
    array = np.zeros(shape = shape)
    return array

#input: int  
def create_identity_numpy_array(order):
    #return a identity numpy array of the defined order
    # creates an identity matrix (order x order)
    array = np.identity(order)
    return array

#input: numpy array
def matrix_cofactor(array):
    #return cofactor matrix of the given array

    rows, cols = array.shape

    # Must be a square matrix  
    if(rows == cols):
        temp_matrix = np.zeros(array.shape)
        minor = np.zeros([array.shape[0] - 1, array.shape[1] - 1])
        for row in range(rows):
            for col in range(cols):
                minor[:row, :col] = array[:row, :col]
                minor[:row, col:] = array[:row, col + 1:]
                minor[row:, :col] = array[row + 1:, :col]
                minor[row:, col:] = array[row + 1:, col + 1:]
                temp_matrix[row][col] = (-1)**(row + col + 2) * np.linalg.det(minor)
        array = temp_matrix
    return array

#Input: (numpy array, int ,numpy array, int , int , int , int , tuple,tuple)
#tuple (x,y)    x,y:int 
def f1(X1,coef1,X2,coef2,seed1,seed2,seed3,shape1,shape2):
    #note: shape is of the forst (x1,x2)
    #return W1 x (X1 ** coef1) + W2 x (X2 ** coef2) +b
    # where W1 is random matrix of shape shape1 with seed1
    # where W2 is random matrix of shape shape2 with seed2
    # where B is a random matrix of comaptible shape with seed3
    # if dimension mismatch occur return -1

    # random matrix of shape = shape1 of seed = seed1
    np.random.seed(seed1)
    W1 = np.random.rand(shape1[0],shape1[1])

    # random matrix of shape = shape2 of seed = seed2
    np.random.seed(seed2)
    W2 = np.random.rand(shape2[0],shape2[1])
    
    # X1 power coef1
    x1_c1 = X1 ** coef1
    # X2 power coef2
    x2_c2 = X2 ** coef2
    
    w1_x1_c1 = np.matmul(W1, x1_c1)
    w2_x2_c2 = np.matmul(W2, x2_c2) 
    
    # if there is mis-match in shape return -1 
    if(w1_x1_c1.shape != w2_x2_c2.shape):
        return -1

    # random matrix of shape = w1_x1_c1.shape or w2_x2_c2.shape since both are equal
    np.random.seed(seed3)
    b = np.random.rand(w1_x1_c1.shape[0], w1_x1_c1.shape[1])

    return w1_x1_c1 + w2_x2_c2 + b



def fill_with_mode(filename, column):
    """
    Fill the missing values(NaN) in a column with the mode of that column
    Args:
        filename: Name of the CSV file.
        column: Name of the column to fill
    Returns:
        df: Pandas DataFrame object.
        (Representing entire data and where 'column' does not contain NaN values)
        (Filled with above mentioned rules)
    """
    # reads the csv file and converts it into pd dataframe object
    df = pd.read_csv(filename)
    # replaces the nan values in every column with the mode value of that column
    df[column].fillna(df[column].mode()[0], inplace = True)
    return df

def fill_with_group_average(df, group, column):
    """
    Fill the missing values(NaN) in column with the mean value of the 
    group the row belongs to.
    The rows are grouped based on the values of another column

    Args:
        df: A pandas DataFrame object representing the data.
        group: The column to group the rows with
        column: Name of the column to fill
    Returns:
        df: Pandas DataFrame object.
        (Representing entire data and where 'column' does not contain NaN values)
        (Filled with above mentioned rules)
    """
    # missing values in a particular column is replaced by the mean value of the group the row belongs
    df[column].fillna(df.groupby(group)[column].transform('mean'), inplace = True)
    return df


def get_rows_greater_than_avg(df, column):
    """
    Return all the rows(with all columns) where the value in a certain 'column'
    is greater than the average value of that column.

    row where row.column > mean(data.column)

    Args:
        df: A pandas DataFrame object representing the data.
        column: Name of the column to fill
    Returns:
        df: Pandas DataFrame object.
    """
    # df has all the rows where value in particular column is greater than average of that column
    df = df[df[column] > df[column].mean()]
    return df