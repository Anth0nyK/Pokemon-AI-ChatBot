#######################################################
#  Initialise NLTK Inference
#######################################################
from nltk.sem import Expression
from nltk.inference import ResolutionProver
read_expr = Expression.fromstring

#######################################################
#  Initialise Knowledgebase. 
#######################################################
import pandas

kb=[]
data = pandas.read_csv('pokemon.csv', header=None)
#print(data[0])
[kb.append(read_expr(row)) for row in data[0]]
# >>> ADD SOME CODES here for checking KB integrity (no contradiction), 
# otherwise show an error message and terminate
expr = None
answer=ResolutionProver().prove(expr, kb, verbose=True)
if answer:
   print('KB Integrity check failed. Please check your kb file.')
   raise SystemExit
else:
   print('KB Integrity check passed.') 
   
#######################################################
#  Initialise AIML agent
#######################################################
import aiml
# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-logic.xml")

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
        answer = kern.respond(userInput)
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        # >> YOU already had some other "if" blocks here from the previous 
        # courseworks which are not shown here.
        
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
               print('It may not be true.') 
               
               expr=read_expr('-'+subject + '(' + object + ')')
               answer=ResolutionProver().prove(expr,kb,verbose=True)
               
               if answer:
                   print('It is incorret.')
               else:
                   print('Sorry I do not know the answer.')
               
               # >> This is not an ideal answer.
               # >> ADD SOME CODES HERE to find if expr is false, then give a
               # definite response: either "Incorrect" or "Sorry I don't know." 
               
               
        elif cmd == 33:
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
                    
        elif cmd == 34:
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
                    
        elif cmd == 35:
            object,subject=params[1].split(' cannot use ')
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
                    print('OK, I will remember that',object,'cant use', subject)
                    kb.append(expr)   
                    
        elif cmd == 36: # if the input pattern is "check that * is *"
            object,subject=params[1].split(' cannot use ')
            expr=read_expr('-' + subject.replace(" ","") + '(' + object + ')')
            answer=ResolutionProver().prove(expr, kb, verbose=True)
            if answer:
               print('Correct.')
            else:
               print('It may not be true.') 
               
               expr=read_expr('-'+subject + '(' + object + ')')
               answer=ResolutionProver().prove(expr,kb,verbose=True)
               
               if answer:
                   print('It is incorret.')
               else:
                   print('Sorry I do not know the answer.')
                   
        elif cmd == 99:
            print("I did not get that, please try again.")
    else:
        print(answer)