#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic chatbot design --- for your own modifications
"""
#######################################################
# Initialise Wikipedia agent
#######################################################
import wikipedia

#######################################################
# Initialise weather agent
#######################################################
import json, requests
#insert your personal OpenWeathermap API key here if you have one, and want to use this feature
APIkey = "5403a1e0442ce1dd18cb1bf7c40e776f" 





#######################################################
#  Initialise NLTK Inference
#######################################################
from nltk.sem import Expression
from nltk.inference import ResolutionProver
read_expr = Expression.fromstring
#######################################################

#######################################################
#  Initialise Knowledgebase. 
#######################################################
import pandas
import sys

kb=[]
data = pandas.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]
# >>> ADD SOME CODES here for checking KB integrity (no contradiction), 
# otherwise show an error message and terminate
expr = None
answer=ResolutionProver().prove(expr, kb, verbose=False)
if answer:
   print('KB Integrity check failed. Please check your kb file.')
   sys.exit()
else:
   print('KB Integrity check passed.') 

#######################################################



#######################################################
#  Initialise spellchecker and regular expressions
#######################################################
from spellchecker import SpellChecker
import re
spell = SpellChecker()

def askYN():
    yes={'yes','y'}
    no={'no','n'}
    
    done = False
    #print(question)
    while not done:
        userChoice = input().lower()
        if userChoice in yes:
            return True
        elif userChoice in no:
            return False
        else:
            print("Please respond in yes or no.")


#######################################################
#  TFIDF and Cosine Similarity 
#######################################################
import csv
import math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#######################################################




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
    
    #old code without using set()
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
    
#######################################################



#######################################################
#  Initialise AIML agent
#######################################################
import aiml
# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
# Use the Kernel's bootstrap() method to initialize the Kernel. The
# optional learnFiles argument is a file (or list of files) to load.
# The optional commands argument is a command (or list of commands)
# to run after the files are loaded.
# The optional brainFile argument specifies a brain file to load.
kern.bootstrap(learnFiles="mybot-logic.xml")

#######################################################




#######################################################
# Welcome user
#######################################################
print("Welcome to this chat bot. Please feel free to ask questions from me!")




#######################################################
# Main loop
#######################################################

while True:
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        #remove the punctuations from userinput with regular expressions
        userInput = re.sub(r'[^\w\s]','',userInput)
        
        #check the spelling
        SCwords = spell.split_words(userInput)
        SCwords2 = [spell.correction(word) for word in SCwords]
        typoFound = False
        #print(SCwords2)
        
        for i in range(len(SCwords)):
            if(SCwords[i]!=SCwords2[i]):
                typoFound = True
        
        if(typoFound == True):
            newUserInput = ' '.join(SCwords2)
            print("Did you mean " + newUserInput + "? (y/n)")
            YesOrNo = askYN()
            if(YesOrNo == True):
                #if yes, get the answer with the corrected userinput
                userInput = newUserInput
                answer = kern.respond(userInput)
            else:
                #if no, get the answer with the userinput
                answer = kern.respond(userInput)
        else:
            answer = kern.respond(userInput)
    
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        
        elif cmd == 1:
            try:
                wSummary = wikipedia.summary(params[1], sentences=3,auto_suggest=False)
                print(wSummary)
            except:
                print("Sorry, I do not know that. Be more specific!")
        
        elif cmd == 2:
            succeeded = False
            api_url = r"http://api.openweathermap.org/data/2.5/weather?q="
            response = requests.get(api_url + params[1] + r"&units=metric&APPID="+APIkey)
            if response.status_code == 200:
                response_json = json.loads(response.content)
                if response_json:
                    t = response_json['main']['temp']
                    tmi = response_json['main']['temp_min']
                    tma = response_json['main']['temp_max']
                    hum = response_json['main']['humidity']
                    wsp = response_json['wind']['speed']
                    wdir = response_json['wind']['deg']
                    conditions = response_json['weather'][0]['description']
                    print("The temperature is", t, "°C, varying between", tmi, "and", tma, "at the moment, humidity is", hum, "%, wind speed ", wsp, "m/s,", conditions)
                    succeeded = True
            if not succeeded:
                print("Sorry, I could not resolve the location you gave me.")
        
# Here are the processing of the new logical component:
        elif cmd == 31: # if input pattern is "I know that * is *"
            object,subject=params[1].split(' is ')
            expr=read_expr(subject + '(' + object + ')')
            # >>> ADD SOME CODES HERE to make sure expr does not contradict 
            # with the KB before appending, otherwise show an error message.
            
            print('Checking if it contradicts with the KB')
            kbTemp = kb.copy()
            kbTemp.append(expr)
            
            answer=ResolutionProver().prove(None, kbTemp, verbose=True)
            if answer:
                print('Sorry, it contradicts with the KB.')
            else:
                existFlag = False
                for element in kb:
                    if(element == expr):
                        print('It already exists in the KB.')
                        existFlag = True
                
                if(existFlag == False):
                    print('OK, I will remember that',object,'is', subject)
                    kb.append(expr)
                
        elif cmd == 32: # if the input pattern is "check that * is *"
            object,subject=params[1].split(' is ')
            expr=read_expr(subject + '(' + object + ')')
            answer=ResolutionProver().prove(expr, kb, verbose=True)
            if answer:
               print('Correct.')
            else:
               #print('It may not be true.') 
               
               print('Checking if it must be false')
               
               expr=read_expr('-'+subject + '(' + object + ')')
               answer=ResolutionProver().prove(expr,kb,verbose=True)
               
               if answer:
                   print('It is incorret.')
               else:
                   print('Sorry I do not know the answer.')
               
               # >> This is not an ideal answer.
               # >> ADD SOME CODES HERE to find if expr is false, then give a
               # definite response: either "Incorrect" or "Sorry I don't know." 
        
        elif cmd == 99:
            #If the bot cannot find an answer in aiml, find it on csv with similarity based search
            #Open the csv file
            file = open("exampleQA.csv")
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
                    
                    
                #print("I did not get that, please try again.")
    else:
        print(answer)