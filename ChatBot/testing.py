# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 13:10:37 2021

@author: Anthony
"""
import csv
import nltk

import matplotlib.pyplot as plt

from skimage import io
import requests
import json

def askYN():
    yes={'yes','y'}
    no={'no','n'}
    
    done = False
    #print(question)
    while not done:
        userChoice = input().lower()
        if userChoice in yes:
            done = True
            return True
        elif userChoice in no:
            done = True
            return False
        else:
            print("Please respond in yes or no.")



#Open the csv file
file = open("pokemon.csv")
#Get the rows from the file with the reader
csvreader = csv.reader(file)

print("Please enter the pokemon name")
try:
    userInput = input("> ")
except (KeyboardInterrupt, EOFError) as e:
    print("Bye!")
 
rows = []
for row in csvreader:
    rows.append(row)

pokemonSim = {}

thePokemon = ""
foundInCSV = False
foundIt = False

for i in range(len(rows)):
    if(userInput.capitalize() == rows[i][0]):
        #pokemonDesc = rows[i][1]
        foundInCSV = True
        foundIt = True
        thePokemon = userInput[0].lower() + userInput[1:]
        #print(pokemonDesc)
        break
    
    
if(foundInCSV == False):
    for i in range(len(rows)):    
        #print(rows[i][0])
        pokemonSim[rows[i][0]] = nltk.edit_distance(userInput.capitalize(),rows[i][0])    
        best_match = min(pokemonSim, key=pokemonSim.get)


    print("Did you mean " + best_match + "? (y/n)")
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
                            print("can evolve to")
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
                            print("Can evolve to")
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
            
            
    if ((foundIt == True) and (succeeded == False)):
        print("This pokemon cannot be found in the wild. You can get it by evolving it.")
        


    

