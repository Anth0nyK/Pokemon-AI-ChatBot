# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:42:40 2021

@author: Anthony
"""
import csv
import math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def getAlluniqueWords(rows, wordsInyourLine):
    
    allWords = []
    
    for i in range(len(rows)):
        wordsInRow = rows[i][0].split()
        [allWords.append(x) for x in wordsInRow if x not in allWords]
    
    [allWords.append(x) for x in wordsInYourLine if x not in allWords]
    
    #print(allWords)
    return allWords


def tf_function(wordsInYourLine):
    TFdictionary = {}
    
    newlist = []
    [newlist.append(x) for x in wordsInYourLine if x not in newlist]
    #print(newlist)
    
    for i in range(len(newlist)):
        counter = 0;
        for j in range(len(wordsInYourLine)):
            if newlist[i] == wordsInYourLine[j]:
                counter = counter + 1
        #Term Frequency, tf = nunber of occurences of a word / total word length of sample
        TFdictionary[newlist[i]] = counter / len(wordsInYourLine)
    
    print("TF", TFdictionary, "\n")
    
    return TFdictionary



def idf_function(allWords, rows, yourLine):
    IDFdictionary = {}
    newRows = []
    newRows.append(yourLine)
    
    for i in range(len(rows)):
        newRows.append(rows[i][0])
        
    NumOfSample = len(newRows)
    
    '''
    for i in range(len(allWords)):
        counter = 0
        for j in range(len(newRows)):
            splitedRow = newRows[i].split()
            for k in range(len(splitedRow)):
                print(splitedRow[k])
                #if splitedRow[k] == allWords[i]:
                    #counter = counter + 1
                    #break
            #IDFdictionary[allWords[i]] = counter
            '''
            
    for i in range(len(allWords)):
        counter = 0
        #print(allWords[i])
        for j in range(len(newRows)):
            theList = list(set(newRows[j].split()))
            for k in range(len(theList)):
                if theList[k] == allWords[i]:
                    counter = counter + 1
        IDFdictionary[allWords[i]] = math.log10(NumOfSample/counter)
    
    print("IDF", IDFdictionary, "\n")

    #print(allWords)

    return IDFdictionary


def tfidf_function(tf,idf,allWords):
    tfidfDict = dict.fromkeys(allWords,0)
    
    for key, value in tfidfDict.items():
        for key2, value2 in tf.items():
            if key2 == key:
                tfidfDict[key] = value2 * idf[key]

    print ("TFIDF",tfidfDict,"\n")
    print("_______________________________________________________________________________________________________________________________")
    return tfidfDict


def cosineSim_function(list1,list2):
    
    array_vec_1 = np.array([list1])
    array_vec_2 = np.array([list2])
    
    return cosine_similarity(array_vec_1,array_vec_2)
    
    




while True:
    userInput = input("Please enter a question: ")
    
    #Open the csv file
    file = open("exampleQA2.csv")
    #Get the rows from the file with the reader
    csvreader = csv.reader(file)
    
    #Define your line here
    yourLine = userInput
    #lowercase the line
    yourLine = yourLine.lower()
    #Split yourLine into words and put them into wordsInYourLine lsit
    wordsInYourLine = yourLine.split()
    
    
    #Create a list called rows and insert the rows into it with for loop
    rows = []
    for row in csvreader:
        rows.append(row)
    
    allwords = getAlluniqueWords(rows, wordsInYourLine)
    print(allwords)
    
    '''
    theRow = rows[0][0]
    
    #Split theRow into words and put them into wordsInRow list
    wordsInRow = theRow.split()
    
    print(wordsInRow)
    print(wordsInYourLine)
    '''
    
    
    idf = idf_function(allwords, rows, yourLine)
    
    #Term Frequency, tf = nunber of occurences of a word / total word length of sample
    #tfYourLine = 
    Yourtf = tf_function(wordsInYourLine)
    
    
    yourTFIDF = tfidf_function(Yourtf, idf, allwords)
    
    NumberOfSampleInCSV = len(rows)
    cosineSimList = [None]*NumberOfSampleInCSV
    
    for i in range(len(rows)):
        wordsIntheLine = rows[i][0].split()
        tfOftheLine = tf_function(wordsIntheLine)
        TFIDFofthLine = tfidf_function(tfOftheLine, idf, allwords)
        #Calculate cosine simularity with "Your TFIDF" and one of the line's TFIDF and save the data in cosineSimList
        cosineSimList[i] = float(cosineSim_function(list(yourTFIDF.values()),list(TFIDFofthLine.values())))

    
    print("")
    print("Cosine Similarity of your line with the line on KB accordingly:")
    print(cosineSimList)
    
    
    #Find the max cosine simularity from the list and get that index to find the corresponding question
    max_value = max(cosineSimList)
    max_index = cosineSimList.index(max_value)
    
    print("\n\n")
    
    #If max value is 0, it means that no similar question is found with your line
    if max_value == 0:
        print("Sorry, we do not have a similar question in the KB :(")
    else:
        print("Your question:", yourLine, ", is similar to:")
        print(rows[max_index][0])
        print("")
        print("Answer:", rows[max_index][1])
    
        file.close()

    

