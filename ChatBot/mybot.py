#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic chatbot design --- for your own modifications
"""
print("Booting...")

# #######################################################
# # Initialise Wikipedia agent
# #######################################################
# import wikipedia
# #######################################################


#######################################################
# Initialise weather agent
#######################################################
import json, requests
#insert your personal OpenWeathermap API key here if you have one, and want to use this feature
APIkey = "5403a1e0442ce1dd18cb1bf7c40e776f" 
#######################################################




#######################################################
#  Initialise speech recognition
#######################################################
import speech_recognition
sr = speech_recognition.Recognizer()

#import keyboard
#######################################################



#######################################################
#  Initialise NLTK Inference
#######################################################
from nltk.sem import Expression
from nltk.inference import ResolutionProver
import nltk
read_expr = Expression.fromstring
#######################################################

#######################################################
#  Initialise Knowledgebase for logical inference 
#######################################################
import pandas

kb=[]
data = pandas.read_csv('pokemonLogic.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]
# >>> ADD SOME CODES here for checking KB integrity (no contradiction), 
# otherwise show an error message and terminate

#prove NULL in the kb
#if there is contradiction, it means that there are problems inside the kb
expr = None
answer=ResolutionProver().prove(expr, kb, verbose=False)
if answer:
   print('KB Integrity check failed. Please check your kb file.')
   raise SystemExit
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
        userChoice = input("> ").lower()
        if userChoice in yes:
            return True
        elif userChoice in no:
            return False
        else:
            print("Please respond in yes or no.")

######################################################



######################################################
#   Image plotter
######################################################
import matplotlib.pyplot as plt
from skimage import io

######################################################



######################################################
#   simpful fuzzy logic
######################################################
from simpful import *

# A simple fuzzy inference system for the tipping problem
# Create a fuzzy system object
FS = FuzzySystem()

S_1 = FuzzySet(points=[[0., 1.],  [93., 0.]], term="low")
S_2 = FuzzySet(points=[[0., 0.], [93., 1.], [186., 0.]], term="normal")
S_3 = FuzzySet(points=[[93., 0.],  [186., 1.]], term="high")
FS.add_linguistic_variable("IV", LinguisticVariable([S_1, S_2, S_3], concept="IV"))

F_1 = FuzzySet(points=[[175., 1.],  [382.5, 0.]], term="low")
F_2 = FuzzySet(points=[[175., 0.], [382.5, 1.], [590., 0.]], term="normal")
F_3 = FuzzySet(points=[[382.5, 0.],  [590., 1.]], term="high")
FS.add_linguistic_variable("BS", LinguisticVariable([F_1, F_2, F_3], concept="BS"))

# Define output crisp values
FS.set_crisp_output_value("bad", 0)
FS.set_crisp_output_value("normal", 33)
FS.set_crisp_output_value("good", 66)
FS.set_crisp_output_value("excellent", 99)

# Define fuzzy rules
R1 = "IF (BS IS high) AND (IV IS high) THEN (Strength IS excellent)"
R2 = "IF (BS IS high) AND (IV IS low) THEN (Strength IS good)"
R3 = "IF (BS IS high) AND (IV IS normal) THEN (Strength IS good)"
R4 = "IF (BS IS low) AND (IV IS high) THEN (Strength IS normal)"
R5 = "IF (BS IS low) AND (IV IS low) THEN (Strength IS bad)"
R6 = "IF (BS IS low) AND (IV IS normal) THEN (Strength IS bad)"
R7 = "IF (BS IS normal) AND (IV IS normal) THEN (Strength IS normal)"
R8 = "IF (BS IS normal) AND (IV IS low) THEN (Strength IS bad)"
R9 = "IF (BS IS normal) AND (IV IS high) THEN (Strength IS good)"
FS.add_rules([R1, R2, R3, R4, R5, R6, R7, R8, R9])
######################################################




#######################################################
#  TFIDF and Cosine Similarity 
#######################################################
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
    
    #print("TF", TFdictionary, "\n")
    
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
                    break
        IDFdictionary[allWords[i]] = math.log10(NumOfSample/counter)
    
    #print("IDF", IDFdictionary, "\n")

    #print(allWords)

    return IDFdictionary


def tfidf_function(tf,idf,allWords):
    tfidfDict = dict.fromkeys(allWords,0)
    
    for key, value in tfidfDict.items():
        for key2, value2 in tf.items():
            if key2 == key:
                tfidfDict[key] = value2 * idf[key]

    #print ("TFIDF",tfidfDict,"\n")
    #print("_______________________________________________________________________________________________________________________________")
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
kern.bootstrap(learnFiles="mybot-aiml.xml")

#######################################################

print()


print("==================================================")
print("  _____   ____  _  ________ _____  ________   __")
print(" |  __ \ / __ \| |/ /  ____|  __ \|  ____\ \ / /")
print(" | |__) | |  | | ' /| |__  | |  | | |__   \ V / ")
print(" |  ___/| |  | |  < |  __| | |  | |  __|   > <  ")
print(" | |    | |__| | . \| |____| |__| | |____ / . \ ")
print(" |_|     \____/|_|\_\______|_____/|______/_/ \_\ ")
print("==================================================")





#######################################################
# Welcome user
#######################################################
print("Welcome Pokemon trainer! Please feel free to ask questions from me!")


#######################################################
# Main loop
#######################################################

while True:
    print("")
    print("This chat bot supports communications in text and voice.")
    print("1 - Text mode")
    print("2 - Voice mode")
    print("Please select 1 or 2:")
    
    mode = ""
    
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    
    if userInput == "1":
        print("Text mode selected")
        mode = "text"
        break
    
    elif userInput == "2":
        print("Voice mode selected")
        mode = "voice"
        break
    
    else:
        print("Please input 1 or 2.")
        
        

while True:
    if mode == "text":
        #get user input
        try:
            userInput = input("> ")
        except (KeyboardInterrupt, EOFError) as e:
            print("Bye!")
            break
    
    elif mode == "voice":
        print("")
        input("When you are ready, press Enter to start talking")

        while True:
            try:
                #use microphone and recognize the audio
                with speech_recognition.Microphone() as mic:
                    sr.adjust_for_ambient_noise(mic, duration=0.2)
                    audio = sr.listen(mic)
                    userInput = None
                    userInput = sr.recognize_google(audio)
                    print(f"Recognized {userInput}")
                    
            except speech_recognition.UnknownValueError():
                sr = speech_recognition.Recognizer()
                continue
            
            if userInput != None:
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
        
        #if there is typo found, ask the user if they meant something
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
        
        # elif cmd == 1:
        #     try:
        #         wSummary = wikipedia.summary(params[1], sentences=3,auto_suggest=False)
        #         print(wSummary)
        #     except:
        #         print("Sorry, I do not know that. Be more specific!")
        
        # elif cmd == 2:
        #     succeeded = False
        #     api_url = r"http://api.openweathermap.org/data/2.5/weather?q="
        #     response = requests.get(api_url + params[1] + r"&units=metric&APPID="+APIkey)
        #     if response.status_code == 200:
        #         response_json = json.loads(response.content)
        #         if response_json:
        #             t = response_json['main']['temp']
        #             tmi = response_json['main']['temp_min']
        #             tma = response_json['main']['temp_max']
        #             hum = response_json['main']['humidity']
        #             wsp = response_json['wind']['speed']
        #             wdir = response_json['wind']['deg']
        #             conditions = response_json['weather'][0]['description']
        #             print("The temperature is", t, "°C, varying between", tmi, "and", tma, "at the moment, humidity is", hum, "%, wind speed ", wsp, "m/s,", conditions)
        #             succeeded = True
        #     if not succeeded:
        #         print("Sorry, I could not resolve the location you gave me.")
                
        elif cmd == 3: #what is a pokemon
            
            pokemonToFind = params[1]
            #Open the csv file
            file = open("pokemon.csv")
            #Get the rows from the file with the reader
            csvreader = csv.reader(file)
            
             
            rows = []
            for row in csvreader:
                rows.append(row)
            
            pokemonSim = {}
            
            thePokemon = ""
            foundInCSV = False
            foundIt = False
            totallyWrong = False
            
            for i in range(len(rows)):
                if(pokemonToFind.capitalize() == rows[i][0]):
                    pokemonDesc = rows[i][1]
                    foundInCSV = True
                    foundIt = True
                    thePokemon = pokemonToFind[0].lower() + pokemonToFind[1:]
                    print(pokemonDesc)
                    break
                
            #if not found in csv directly, it means that there could be typo of the pokemon name
            #Pokemon names are not in dictionary, so need to have an additional doc to know the correct pokemon names
            #use nltk edit distance to find the best match
            if(foundInCSV == False):
                for i in range(len(rows)):    
                    #print(rows[i][0])
                    pokemonSim[rows[i][0]] = nltk.edit_distance(pokemonToFind.capitalize(),rows[i][0])    
                    #print(pokemonSim)
                    best_match = min(pokemonSim, key=pokemonSim.get)
                    
                totallyWrong = False
                if(int(pokemonSim[best_match]) > len(best_match)):
                    totallyWrong = True
                    
                
                #if the pokemon name is not totally wrong, ask the user if the best match is what they want to find
                if(totallyWrong == False):
                    
                    print("Did you mean the Pokemon " + best_match + "? (y/n)")
                    YesOrNo = askYN()
                    if(YesOrNo == True):
                        #if yes
                        for i in range(len(rows)):
                            if(best_match.capitalize() == rows[i][0]):
                                pokemonDesc = rows[i][1]
                                thePokemon = best_match[0].lower() + best_match[1:]
                                #print("!!!!!!"+thePokemon)
                                print(pokemonDesc)
                                foundIt = True
                                break
                    else:
                        #if no
                        print("Sorry, we cannot find that pokemon.")
                        
                else:
                    print("Sorry, we do not understand what did you mean.")
            
            
            #if found the needed pokemon for the user, call the api to get the pokemon image
            if(foundIt):  
                #print("image part")
                succeeded = False
                api_url = r"https://pokeapi.co/api/v2/pokemon/"
                response = requests.get(api_url + thePokemon)
                #print("!!!!" + thePokemon)
                
                if response.status_code == 200:
                    response_json = json.loads(response.content)
                    if response_json:
                        image = response_json['sprites']['front_default']
                        #print(image)
                        succeeded = True
                        image = io.imread(image)
                        plt.imshow(image)
                        plt.show()
                if not succeeded:
                    print("Sorry, I could not find an image of that Pokemon.")
            
            
            
            file.close()
            
            #check if the user is asking what is the strongest pokemon,etc.
            #as they both share the same "what is *"
            #use the code from the tfidf part here and screen the QA csv
            if(totallyWrong == True):
                file = open("QA.csv")
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
                #print(allwords)
                
                
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
            
            
        elif cmd == 4: #where to find pokemon
            pokemonToFind = params[1].capitalize() 
            #Open the csv file
            file = open("pokemon.csv")
            #Get the rows from the file with the reader
            csvreader = csv.reader(file)
        
            rows = []
            for row in csvreader:
                rows.append(row)
            
            pokemonSim = {}
            
            thePokemon = ""
            foundInCSV = False
            foundIt = False
            
            for i in range(len(rows)):
                if(pokemonToFind.capitalize() == rows[i][0]):
                    #pokemonDesc = rows[i][1]
                    foundInCSV = True
                    foundIt = True
                    thePokemon = pokemonToFind[0].lower() + pokemonToFind[1:]
                    #print(pokemonDesc)
                    break
                
                
            if(foundInCSV == False):
                for i in range(len(rows)):    
                    #print(rows[i][0])
                    pokemonSim[rows[i][0]] = nltk.edit_distance(pokemonToFind.capitalize(),rows[i][0])    
                    best_match = min(pokemonSim, key=pokemonSim.get)


                totallyWrong = False
                if(len(best_match) == int(pokemonSim[best_match])):
                    totallyWrong = True

                    
                if(totallyWrong == False):
                        
                    print("Did you mean the Pokemon " + best_match + "? (y/n)")
                    YesOrNo = askYN()
                    if(YesOrNo == True):
                        #if yes
                        for i in range(len(rows)):
                            if(best_match.capitalize() == rows[i][0]):
                                #pokemonDesc = rows[i][1]
                                thePokemon = best_match[0].lower() + best_match[1:]
                                #print("!!!!!!"+thePokemon)
                                #print(pokemonDesc)
                                foundIt = True
                                break
                    else:
                        #if no
                        print("Sorry, we cannot find that pokemon.")
                
                else:
                    print("Sorry, we do not understand what did you mean.")

            #if found the pokemon which is what the uesr want to know
            #get the encounter locations of the pokemon for the user from the api
            if(foundIt):  
                succeeded = False
                api_url = r"https://pokeapi.co/api/v2/pokemon/"
                response = requests.get(api_url + thePokemon + "/encounters")
                if response.status_code == 200:
                    response_json = json.loads(response.content)
                    if response_json:
                        arrayLength = len(response_json)
                        #image = response_json['sprites']['front_default']
                        locationList = []
                        
                        for i in range(arrayLength):
                            locationList.append(response_json[i]["location_area"]["name"].replace('-', ' '))
                        if(arrayLength > 1):
                            print("\n" + "You can find this Pokemon in these area:")
                        else:
                            print("\n" + "You can only find this Pokemon in this area:")
                        for i in range(arrayLength):
                            print(locationList[i], end = '')
                            if(i == arrayLength-1):
                                print(".")
                            else:
                                print(", ", end = '')
                        #print(image)
                        #print("!!!!!" + str(location))
                        succeeded = True
                        
                        #image = io.imread(image)
                        #plt.imshow(image)
                        #plt.show()
                if ((foundIt == True) and (succeeded == False)):
                    print("This pokemon cannot be found in the wild. You can get it by evolving it.")
                        
            file.close()
        
            
            
        elif cmd == 5: #what can pokemon evolve into
            pokemonToFind = params[1].capitalize()
            #Open the csv file
            file = open("pokemon.csv")
            #Get the rows from the file with the reader
            csvreader = csv.reader(file)
            
            rows = []
            for row in csvreader:
                rows.append(row)
            
            pokemonSim = {}
            
            thePokemon = ""
            foundInCSV = False
            foundIt = False
            
            for i in range(len(rows)):
                if(pokemonToFind.capitalize() == rows[i][0]):
                    #pokemonDesc = rows[i][1]
                    foundInCSV = True
                    foundIt = True
                    thePokemon = pokemonToFind[0].lower() + pokemonToFind[1:]
                    #print(pokemonDesc)
                    break
                
                
            if(foundInCSV == False):
                for i in range(len(rows)):    
                    #print(rows[i][0])
                    pokemonSim[rows[i][0]] = nltk.edit_distance(pokemonToFind.capitalize(),rows[i][0])    
                    best_match = min(pokemonSim, key=pokemonSim.get)
            
                totallyWrong = False
                if(len(best_match) == int(pokemonSim[best_match])):
                    totallyWrong = True
                    
                if(totallyWrong == False):
                
                    print("Did you mean the Pokmeon " + best_match + "? (y/n)")
                    YesOrNo = askYN()
                    if(YesOrNo == True):
                        #if yes
                        for i in range(len(rows)):
                            if(best_match.capitalize() == rows[i][0]):
                                #pokemonDesc = rows[i][1]
                                thePokemon = best_match[0].lower() + best_match[1:]
                                #print("!!!!!!"+thePokemon)
                                #print(pokemonDesc)
                                foundIt = True
                                break
                    else:
                        #if no
                        print("Sorry, we cannot find that pokemon.")
                else:
                    print("Sorry, we do not understand what did you mean.")
            
            #if found the pokemon which is what the user want
            #call the api and get the evolvution tree of that pokemon
            if(foundIt):  
                succeeded = False
                api_url = r"https://pokeapi.co/api/v2/pokemon-species/"
                response = requests.get(api_url + thePokemon)
                if response.status_code == 200:
                    response_json = json.loads(response.content)
                    if response_json:
                        
                        
                        evoChain = response_json['evolution_chain']['url']
                        #print(evoChain)
                        
                        api_url2 = evoChain
                        response2 = requests.get(api_url2)
                        if response2.status_code == 200:
                            response_json2 = json.loads(response2.content)
                            if response_json2:
                                haveFirst = False
                                firstUrl = response_json2['chain']['species']['url']
                                firstID = firstUrl[42:]
                                firstID = firstID[:-1]
                                if(int(firstID) <= 151):
                                    firstForm = response_json2['chain']['species']['name']
                                    print(firstForm.capitalize())
                                    haveFirst = True
                                    
                                try:
                                    middleLength = len(response_json2['chain']['evolves_to'])
                                    if(middleLength != 0 and haveFirst == True):
                                        print(" ")
                                        print("can evolve into")
                                        print(" ")
                                    for i in range(middleLength):
                                        middleUrl = response_json2['chain']['evolves_to'][i]['species']['url']
                                        middleID = middleUrl[42:]
                                        middleID = middleID[:-1]
                                        if(int(middleID) <= 151):
                                            middleForm = response_json2['chain']['evolves_to'][i]['species']['name']
                                            print(middleForm.capitalize())
                                            
                                    finalLength = len(response_json2['chain']['evolves_to'][0]['evolves_to'])
                                    #print(finalLength)
                                    if(finalLength != 0):
                                        print(" ")
                                        print("can evolve into")
                                        print(" ")
                                    for j in range(finalLength):
                                        finalUrl = response_json2['chain']['evolves_to'][0]['evolves_to'][j]['species']['url']
                                        finalID = finalUrl[42:]
                                        finalID = finalID[:-1]
                                        if(int(finalID) <= 151):
                                            finalForm = response_json2['chain']['evolves_to'][0]['evolves_to'][j]['species']['name']
                                            print(finalForm.capitalize())
            
                                except:
                                    print("")
                                    
                                succeeded = True
                        
                #if cannot find the pokemon
                if ((foundIt == True) and (succeeded == False)):
                    print("This pokemon cannot be found in the wild. You can get it by evolving it.")
            
            
            
            
            
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
            expr=read_expr(subject.replace(" ","") + '(' + object + ')')
            answer=ResolutionProver().prove(expr, kb, verbose=True)
            if answer:
               print('Correct.')
            else:
               print('It may not be true.') 
               
               expr=read_expr('-'+subject.replace(" ","") + '(' + object + ')')
               answer=ResolutionProver().prove(expr,kb,verbose=True)
               
               if answer:
                   print('It is incorret.')
               else:
                   print('Sorry I do not know the answer.')
               
               # >> This is not an ideal answer.
               # >> ADD SOME CODES HERE to find if expr is false, then give a
               # definite response: either "Incorrect" or "Sorry I don't know." 
               
               
        elif cmd == 33: #I know that * is not *
            object,subject=params[1].split(' is not ')
            expr=read_expr('-' + subject + '(' + object + ')')
            
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
                    print('OK, I will remember that',object,'is not', subject)
                    kb.append(expr)
                    
        elif cmd == 34: #I know that * can use *
            object,subject=params[1].split(' can use ')
            expr=read_expr(subject.replace(" ", "") + '(' + object + ')')
            
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
                    print('OK, I will remember that',object,'can use', subject)
                    kb.append(expr)
                    
        elif cmd == 35: #I know that * cannot use *
            object,subject=params[1].split(' cannot use ')
            expr=read_expr("-"+subject.replace(" ", "") + '(' + object + ')')
            
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
                    print('OK, I will remember that',object,'cant use', subject)
                    kb.append(expr)   
                    
        elif cmd == 36: # if the input pattern is "check that * cannot use *"
            object,subject=params[1].split(' cannot use ')
            expr=read_expr('-' + subject.replace(" ","") + '(' + object + ')')
            answer=ResolutionProver().prove(expr, kb, verbose=True)
            if answer:
               print('Correct.')
            else:
               print('It may not be true.') 
               
               expr=read_expr(subject.replace(" ","") + '(' + object + ')')
               answer=ResolutionProver().prove(expr,kb,verbose=True)
               
               if answer:
                   print('It is incorret.') 
               else:
                   print('Sorry I do not know the answer.')
                   
                   
        elif cmd == 37: # Fuzzy logic check that this pokemon is strong
            while(True):
                print('Please input the sum of individual values (IVs) of the Pokemon (0 - 186): ')
                sumOfIV = input()
                if (int(sumOfIV) <= 186 and int(sumOfIV) >= 0):
                    break
                
            while(True):
                print('Please input the base states of the Pokemon (175 - 590): ')
                baseStates = input()
                if (int(baseStates) >= 175 and int(baseStates) <= 590):
                    break
            
            # Set antecedents values
            FS.set_variable("IV", sumOfIV)
            FS.set_variable("BS", baseStates)
            
            
            FS.plot_variable("IV")
            FS.plot_variable("BS")
            print('The strength of your Pokemon (0 - Worst ,33 - Normal, 66 - Good, 99 - Best)')
            # Perform Sugeno inference and print output
            print(FS.Sugeno_inference(["Strength"]))
                
                   
        #go into this if questions are not in aiml
        elif cmd == 99: #do tfidf and cosine similarity here
            #If the bot cannot find an answer in aiml, find it on csv with similarity based search
            
            #Open the csv file
            file = open("QA.csv")
            #file = open("forTesting.csv")
            
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
            print("Cosine Similarity of your question with the qestions on KB:")
            print(cosineSimList)
            
            
            #Find the max cosine simularity from the list and get that index to find the corresponding question
            max_value = max(cosineSimList)
            max_index = cosineSimList.index(max_value)
            
            print("\n")
            
            #If max value is 0, it means that no similar question is found with your line
            if max_value == 0:
                print("Sorry, we cannot find a similar question in the KB.")
            else:
                print("Your question:", yourLine, ", is similar to:")
                print(rows[max_index][0])
                print("")
                print("Answer:", rows[max_index][1])
            
                file.close()
                    
                    
                #print("I did not get that, please try again.")
    else:
        print(answer)