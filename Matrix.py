#Matrix class
#Bram Faes

import numpy as np
from random import *

class matrix(object):

    #Initialize the matrix
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = np.random.rand(rows,cols)
    #Set every element in the 2D matrix to a random number
    def randomize(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = self.data[i][j] * 2 - 1

    #Scale every element in matrix by a single number
    def scalar(self, n):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] *= n
    #Scale every element in matrix by another matrix elementwise
    def scalarMatrix(self, m):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] *= m.data[i][j]

    #Add a single number to every element in a matrix
    def add(self, n):
        if isinstance(n, (int, float)):
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n
        else:
            # assume n is a matrix
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n.data[i][j]

    #Add a single number from other matrix to every element in a matrix elementwise
    def addMtrx(self, m):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] += m.data[i][j]

    #Apply a function to every element of the matrix
    def map1(self, fn):
        for i in range(self.rows):
            for j in range(self.cols):
                val = self.data[i][j]
                self.data[i][j] = fn(val)

#------------------------------------------------------------------------------------

def arrToMatrix(a):
    m = matrix(len(a),1)
    for i in range(len(a)):
        m.data[i][0] = a[i]
    return m

def mtrxToArr(o):
    arr = []
    for i in range(o.rows):
        for j in range(o.cols):
            arr.append(o.data[i][j])
    return arr

def subMtrx(m, n):
    new = matrix(m.rows, m.cols)
    for i in range(new.rows):
        for j in range(new.cols):
            new.data[i][j] = m.data[i][j] - n.data[i][j]
    return new

#Transpose a matrix (e.g. from 2x3 to 3x2)
def transpose(m):
    n = matrix(m.cols, m.rows)
    for i in range(m.rows):
        for j in range(m.cols):
            n.data[j][i] = m.data[i][j]
    return n

def map2(m, fn):
    new = matrix(m.rows, m.cols)
    for i in range(m.rows):
        for j in range(m.cols):
            val = m.data[i][j]
            new.data[i][j] = fn(val)
    return new

def sigmoid(x):
    return (1/(1+np.exp(-x)))


#Matrix product of two matrixes (same amount of A-cols and B-rows)
def mtrxMultiplyStatic(m, n):
    if m.cols == n.rows:
        c = matrix(m.rows, n.cols)
        for i in range(c.rows):
            for j in range(c.cols):
                temp = 0
                for k in range(m.cols):
                    temp += m.data[i][k] * n.data[k][j]
                c.data[i][j] = temp
        return c
    else:
       print("not possible")

#------------------------------------------------------------------------------------
