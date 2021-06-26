#from .readWorkflow import lstConnection
import os, sys , inspect

sys.path.append(os.getcwd())
from VGLGui.readWorkflow import *

# IMPORTING METHODS FROM VISIONGL
#sys.path.insert(0,'VisionGL/src/py')
#from VisionGL.src.py import benchmark_clnd

#Show info
def procShowInfo():
    for vGlyph in lstGlyph:
        print("Library:", vGlyph.library, "Function:", vGlyph.func, "Localhost:", vGlyph.localhost, "Glyph_Id:", vGlyph.glyph_id, 
            "Position_Line:", vGlyph.glyph_x, "Position_Column:", vGlyph.glyph_y)#, "Parameters:", vGlyph.lst_par)

        #Shows the list of glyph inputs
        for vGlyphIn in vGlyph.lst_input:
            print("Glyph_Id:", vGlyph.glyph_id, "Glyph_In:", vGlyphIn)

        #Shows the list of glyph outputs
        for vGlyphOut in vGlyph.lst_output:
            print("Glyph_Id:", vGlyph.glyph_id, "Glyph_Out:", vGlyphOut)

    # Shows the content of the Connections
    for vConnection in lstConnection:
        print("Conexão:", vConnection.type, "Glyph_Output_Id:", vConnection.output_glyph_id, "Glyph_Output_Varname:", vConnection.output_varname,
            "Glyph_Input_Id:", vConnection.input_glyph_id, "Glyph_Input_Varname:", vConnection.input_varname)

# Program execution

# Reading the workflow file and loads into memory all glyphs and connections
fileRead(lstGlyph)

#Update the status of glyph entries
for vGlyph in lstGlyph:

    if vGlyph.getGlyphReady() and vGlyph.getGlyphDone() == False:

        #Glyph execute
        # xxxxxxxxx

        #Sets all glyph outputs as executed 
        vGlyph.setGlyphOutputAll(True)

        #Sets all glyph outputs as executed
        vGlyph.setGlyphInputAll(True)

        #Defines that the glyph was executed
        vGlyph.setGlyphDone(True)

# Shows the content of the Glyphs
procShowInfo()


