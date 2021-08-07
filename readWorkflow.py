# Objective: read VGLGui workflow file and load content into memory
# File type: structure.txt

#
import re
import os
import string
from collections import defaultdict

class Error (Exception): #classe para tratar uma execeção definida pelo usuário
    pass
    '''
        FALTA OS AJUSTES PARA SAÍDA DA CLASSE, POREM SÓ COM A FUNÇÃO 'raise' JÁ FUNCIONA
    '''
# Structure for storing Glyphs in memory
# Glyph represents a function
class objGlyph(object):
        
    #Glyph:[Library]:comment::localhost:[Glyph_ID]:[Glyph_X]:[Glyph_Y]:: -[var_str] '[var_str_value]' -[var_num] [var_num_value]    
    def __init__(self, vlibrary, vfunc, vlocalhost, vglyph_id, vglyph_x, vglyph_y):       
        self.library = vlibrary                 #library name (Ex: VisionGL)
        self.func = vfunc                       #function
        self.localhost = vlocalhost             #folder where the image file or library function is located
        self.glyph_id = vglyph_id               #glyph identifier code
        self.glyph_x = vglyph_x                 #numerical coordinate of the glyph's linear position on the screen 
        self.glyph_y = vglyph_y                 #numerical coordinate of the column position of the Glyph on the screen
        
        # Rule7: Glyphs have READY (ready to run) and DONE (executed) status, both status start being FALSE
        self.ready = False                      #TRUE = glyph is ready to run
        self.done = False                       #TRUE = glyph was executed

        self.lst_par = []                       #parameter list

        # Rule1: Glyphs correspond to vertices in a graph and represent a function from the VisionGL library
        #           They have a list of input and output parameters.
        #           Each input is linked to a single output of another glyph.
        #           Each output can be connected to more than one input from another glyph.
        self.lst_input = []                     #glyph input list
        self.lst_output = []                    #glyph output list

    #Add glyph parameter function
    def funcGlyphAddPar (self, vGlyphPar):
        self.lst_par.append(vGlyphPar)

    #Add glyph input function
    def funcGlyphAddIn (self, vGlyphIn):
        self.lst_input.append(vGlyphIn)

    #Add glyph output function 
    def funcGlyphAddOut (self, vGlyphOut):
        self.lst_output.append(vGlyphOut)

    # Return Glyph Ready status
    def getGlyphReady(self):
        return self.ready

    # Rule8: Glyphs have a list of entries. When all entries are READY=TRUE, the glyph changes status to READY=TRUE (function ready to run)
    def setGlyphReady(self, status):

        vGlyphReady = status

        #Identifies if all glyph entries were used
        if vGlyphReady == True and len(self.lst_input) > 0:
            
            #If there is an entry without using
            for vGlyphIn in self.lst_input:            
                if vGlyphIn.getStatus() == False:
                    vGlyphReady = False
                    self.ready = False
                    exit    
    
            #If all inputs were used
            if vGlyphReady:
                self.ready = vGlyphReady
        else:
            self.ready = vGlyphReady

    # Rule10: Glyph becomes DONE = TRUE after its execution
    #         Assign done to glyph
    def setGlyphDone(self, status):
        self.done = status

    #Return Done status
    def getGlyphDone(self):
        return self.done

    #Assign ready to glyph inputs
    def setGlyphDoneAllInput(self, status):
        for i, vGlyphIn in enumerate(self.lst_input):
           self.lst_input[i].setGlyphInput(vGlyphIn, status)

    # Rule6: Edges whose source glyph has already been executed, and which therefore already had their image generated, have READY=TRUE (image ready to be processed).
    #        Reading the image from another glyph does not change this status.
    #        Set READY = TRUE to glyph input and READY = TRUE to glyph 
    def setGlyphReadyInput(self, status, vinput_varname):
        for i, vGlyphIn in enumerate(self.lst_input):
           if self.lst_input[i].namein == vinput_varname:
               self.lst_input[i].ready = True
               self.setGlyphReady()
               break

# Structure for storing Parameters in memory
class objGlyphParameters(object):

    def __init__(self, namepar, valuepar):
        self.name = namepar      #variable name
        self.value = valuepar    #variable value

    def getName(self):
        return self.name

    def getValue(self):
        return self.value

# Structure for storing Glyphs input list in memory
class objGlyphInput(object):

    def __init__(self, namein, statusin):
        self.namein = namein         #glyph input name
        self.done = statusin     #glyph input status

    def getStatus(self):
        return self.statusin

    #Assign status to glyph output
    def setGlyphInput(self, status):
        self.done = status


# Structure for storing Glyphs output list in memory
class objGlyphOutput(object):

    def __init__(self, nameout, statusout):
        self.nameout = nameout      #glyph output name
        self.statusout = statusout  #glyph output status

    #Assign status to glyph output
    def setGlyphOutput(self, status):
        self.statusout = status

# Structure for storing Connections in memory
# Images are stored on edges (connections between Glyphs)
class objConnection(object):

    #NodeConnection:data:[output_Glyph_ID]:[output_varname]:[input_Glyph_ID]:[input_varname]        
    def __init__(self, vtype, voutput_glyph_id, voutput_varname, vinput_glyph_id, vinput_varname):       
        self.type = vtype                           #type 'data', 'controle' 
        self.output_glyph_id = voutput_glyph_id     #glyph identifier code output
        self.output_varname = voutput_varname       #variable name output
        self.input_glyph_id = vinput_glyph_id       #glyph identifier code input
        self.input_varname = vinput_varname         #variable name input
        self.image = None                           #image
        self.ready = False                          #False = unread or unexecuted image; True = image read or executed

    # Rule5: Each edge has an image stored
    #        Assign image to Connection
    def setImageConnection(self, img):
        self.image = img

    #Assign image to Connection
    def setReadyConnection(self, img):
        self.ready = True

    def getReadyConnection(self):
        return self.ready

#Create the inputs and outputs for the glyph
def procCreateGlyphInOut():

    for vConnection in lstConnection:
        
        for i, vGlyph in enumerate(lstGlyph):

            # Create the input for the glyph
            if vConnection.input_varname != '\n' and vGlyph.glyph_id == vConnection.input_glyph_id:
                vGlyphIn = objGlyphInput(vConnection.input_varname, False)
                lstGlyph[i].funcGlyphAddIn (vGlyphIn)

            # Create the output for the glyph   
            if vConnection.output_varname != '\n' and vGlyph.glyph_id == vConnection.output_glyph_id:
                vGlyphOut = objGlyphInput(vConnection.output_varname, False)
                lstGlyph[i].funcGlyphAddOut (vGlyphOut)

    #Rule11: Source glyph is already created with READY = TRUE. 
    #        After creating NodeConnections, the Glyph that has no input will be considered of the SOURCE type and 
    #        will have READY = TRUE (ready for execution)
    for i, vGlyph in enumerate(lstGlyph):

       if len(vGlyph.lst_input) == 0:
           lstGlyph[i].setGlyphReady(True)
           
#Identifies and Creates the parameters of the Glyph
def procCreateGlyphParameters(vGlyph, vParameters, count):
    try:

        #Identifies the parameters
        #:: -[var_str] '[var_str_value]' -[var_num] [var_num_value]
        contentGlyPar = []               #clears the glyph parameter list
        lstParAux = []                   #auxiliary parameter list

        contentGlyPar = vParameters

        for vpar in contentGlyPar:
            if vpar != '' and vpar != '\n':

                vGlyphPar = objGlyphParameters 
                vpar = vpar.replace("\n", '') 

                #Differentiates parameter name and value
                if vpar[0] == '\'' or vpar.isdigit():
                    vGlyphPar = objGlyphParameters('Value', vpar.replace("'", '')) 

                if vpar[0] == "-":             
                    if vpar[1].isdigit():
                        vGlyphPar = objGlyphParameters('Value', vpar.replace("-", ''))
                    else:
                        vGlyphPar = objGlyphParameters('Name', vpar.replace('-', ''))

                #Temporary list to differentiate parameters and their values
                lstParAux.append(vGlyphPar)

        #Creates the parameters of the Glyph
        for i, vParAux in enumerate(lstParAux):
            
            vParType = vParAux.getName()
            vParValue = vParAux.getValue()
            
            vParTypeNext = ''
            vParValueNext = ''

            #If you don't have the next parameter to include
            if i < (len(lstParAux)-1):
                vParTypeNext = lstParAux[i+1].getName()
                vParValueNext = lstParAux[i+1].getValue()
            
            # A parameter name followed by another parameter name. Write the parameter because it will have no value. Example: -wh -hw -dd
            if vParType == 'Name' and (vParTypeNext == 'Name' or (vParTypeNext == '' and vParType != 'Value')):
                vGlyphPar = objGlyphParameters(vParValue, '')
                vGlyph.funcGlyphAddPar(vGlyphPar)

            # A parameter name followed by a value. Write the parameter with its value
            if vParType == 'Name' and vParTypeNext == 'Value':
                vGlyphPar = objGlyphParameters(vParValue, vParValueNext)
                vGlyph.funcGlyphAddPar(vGlyphPar)

    except IndexError as d: #rule102 - Variable not found
        print("Non-standard information in the Parameter declaration"," \nLine",{count}, "{d}")
    except ValueError as s: #rule103 - Error in defined Parameters coordinates (not integer or out of bounds)
        print("Non-standard information in the Parameter declaration","\nLine",{count} , "{s}")


# Create Glyph
def procCreateGlyph(contentGly, count):
    try:
        
        vBlib = ''
        vFunc = ''
        vLoc = ''
        vIdGlyh = ''
        vPosX = ''
        vPosY = ''            
        vGlyphPar = ''

        if len(contentGly) == 8:  #Image Input/Outpu type Glyph
            vBlib = contentGly[1]
            vFunc = contentGly[2]
            vLoc = contentGly[3]
            vIdGlyh = contentGly[4]
            vPosX = contentGly[5]
            vPosY = contentGly[6]            
            vGlyphPar = contentGly[7].split(' ')
        elif len(contentGly) > 9: #Image type parameter
            vBlib = contentGly[1]
            vFunc = contentGly[2]
            vLoc = contentGly[4]
            vIdGlyh = contentGly[5]
            vPosX = contentGly[6]
            vPosY = contentGly[7]
            vGlyphPar = contentGly[9].split(' ')            

        vGlyph = objGlyph(vBlib, vFunc, vLoc, vIdGlyh, vPosX, vPosY)

        #Creates the parameters of the Glyph
        procCreateGlyphParameters(vGlyph, vGlyphPar, count)                    

        #rule104 - Invalid screen position or exceeds dimensions to be defined by file
        if (int(contentGly[6]) or int(contentGly[7])) > 100000 or (int(contentGly[6]) or int(contentGly[7])) < 0:
            raise Error("Glyph position on screen in error,", " check the line: ",{count}) 

        #Create the Glyph
        lstGlyph.append(vGlyph)

    except IndexError as d: #rule102 - Variable not found
        print("Non-standard information in the Glyph declaration"," \nLine",{count}, "{d}")
    except ValueError as s: #rule103 - Error in defined glyph coordinates (not integer or out of bounds)
        print("Non-standard information in the Glyph declaration","\nLine",{count} , "{s}")

#Creates the connections of the workflow file
def procCreateConnection(contentCon, count):
    try:
        #NodeConnection:data:[output_Glyph_ID]:[output_varname]:[input_Glyph_ID]:[input_varname]        
        vConnection = objConnection(contentCon[1], contentCon[2], contentCon[3].replace('\n',''), contentCon[4], contentCon[5].replace('\n',''))
        lstConnection.append(vConnection)           

        #rule105 - Invalid Glyph Id
        try:
            if int(contentCon[2])  <0 or int(contentCon[4]) < 0:
                raise Error("Invalid glyph id on line: ",{count})
        except ValueError:
            print("Invalid Connection Creation Values." , " check the line: ",{count})

    except IndexError as f: #rule 102 - Variable not found
        print("Connections indices not found",{f},"on line ",{count}," of the file")

# File to be read
vfile = 'dataVglGui.wksp'

lstGlyph = []                   #List to store Glyphs
lstGlyphPar = []                #List to store Glyphs Parameters
lstConnection = []              #List to store Connections
lstGlyphIn = []                 #List to store Glyphs Inputs
lstGlyphOut = []                #List to store Glyphs Outputs

vGlyph = objGlyph               #Glyph in memory 
vGlyphPar = objGlyphParameters  #Glyph parameters in memory
vGlyphIn = objGlyphInput        #Glyph input in memory
vGlyphOut = objGlyphOutput      #Glyph output in memory
vConnection = objConnection     #Connection in memory

# Method for reading the workflow file
def fileRead(lstGlyph, lstConnection):
    try:
        if os.path.isfile(vfile):

            count = 0 #line counter

            # Opens the workflow file
            file1 = open(vfile,"r")
            for line in file1:

                count +=1   #line counter

                #Extracts the contents of the workflow file line in a list separated by the information between the ":" character and create Glyph
                if ('glyph:' in line.lower()) or ('extport:' in line.lower()):
                    procCreateGlyph(line.split(':'), count)

                # Rule4: Edges are Connections between Glyphs and represent the image to be processed
                #         Creates the connections of the workflow file
                if 'nodeconnection:' in line.lower():
                    procCreateConnection(line.split(':'), count)

            file1.close()

            # Rule11: Source glyph is already created with READY = TRUE. 
            # Create inputs and outputs of the Glyph
            procCreateGlyphInOut()
            
    except UnboundLocalError: #rule101 - File not found
        print("File not found.")