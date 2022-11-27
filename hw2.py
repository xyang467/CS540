import sys
import math
import string
from math import *


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    X = dict.fromkeys(string.ascii_uppercase, 0)
    with open (filename,encoding='utf-8') as f:
        for line in f:
            for char in line.strip():
                if char.isalpha():
                    letter = char.upper()
                    X[letter] += 1
    f.close()
    return X
X = shred("letter.txt")
print("Q1")
for key, value in X.items():
    print(key, value)
    
print("Q2")
e,s = get_parameter_vectors()
print(format(list(X.values())[0]*log(e[0]),'.4f'))
print(format(list(X.values())[0]*log(s[0]),'.4f'))

print("Q3")
Fe = log(0.6) + sum([log(e[i])*list(X.values())[i] for i in range(26)])
Fs = log(0.4) + sum([log(s[i])*list(X.values())[i] for i in range(26)])
print(format(Fe, '.4f'))
print(format(Fs, '.4f'))

print("Q4")
if Fs-Fe >=100:
    p = 0
elif Fs-Fe <=-100:
    p = 1
else:
    p = 1/(1+exp(Fs-Fe))
print(format(p, '.4f'))
