'''
This util file includes the function for raw data preprocessing and artifical data synthesis
'''
import numpy as np
from scipy import interpolate
from random import choice
import random

def Normalization(InputVector):
    '''
    function description:
        InputVector shape: [1,n], where n is the number of element spectrum frequency
        Normalization:
            X = ( X - Xmin) / ( Xmax - Xmin )
    '''
    OutputVector = (InputVector-InputVector.min()) / (InputVector.max() - InputVector.min())
    return OutputVector

def Linear1dInterpolate(InputVector):
    '''
    function description:
        InputVector shape: [2,n] : dismension 0 -> spectrum index || dismension 1 -> spectrum value
        InputVector: experiment data sample
        Linear1dInterpolate:
            Match the format of experiment data samples and theoretical data samples
    '''
    F = interpolate.interp1d(InputVector[0], InputVector[1], kind='linear')
    NewIndex = np.linspace(3, 38, 3501)
    UpdatedVector = F(NewIndex)
    return UpdatedVector

def DataSynthesis(E,T):
    '''
    function description:
        E : a list includes all positive experiment data samples
        T : a list includes all negative theoretical data samples
        **************All the data samples in E or T should be normalized*****************
        DataSynthesis:
            Create aritifical synthesis positive data samples like below:
                Xac = ei + e * ti, where e is a random number belonging to (0,1)
    '''
    ei = choice(E)
    ti = choice(T)
    e = random.uniform(0, 1)
    Xac = ei + e * ti
    return Xac