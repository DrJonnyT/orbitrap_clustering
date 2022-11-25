# -*- coding: utf-8 -*-
import re

class chemform:
    """
    A class for chemical formulae
    """
    def __init__(self, formula):
    
        try:
            self.C = int(re.findall(r'C(\d+)',formula)[0])
        except:
            self.C = len(re.findall(r'C',formula))
        
        try:
            self.H = int(re.findall(r'H(\d+)',formula)[0])
        except:
            self.H = len(re.findall(r'H',formula))
    
        try:
            self.O = int(re.findall(r'O(\d+)',formula)[0])
        except:
            self.O = len(re.findall(r'O',formula))
        
        try:
            self.S = int(re.findall(r'S(\d+)',formula)[0])
        except:
            self.S = len(re.findall(r'S',formula))
        
        try:
            self.N = int(re.findall(r'N(\d+)',formula)[0])
        except:
            self.N = len(re.findall(r'N',formula))
            
    def classify(self):
        """

        Returns
        -------
        classification : str
            A string containing letters for the elements contained in the chemform.

        """
        
        classification = ''
        
        for letter in 'CHONS':
            if getattr(self,letter) > 0:
                classification += letter
        
        return classification
        
        # if(self.C>0 and self.H>0 and self.O>0):
        #     if self.S>0:
        #         if self.N>0:
        #             return 'CHNOS'
        #         else:
        #             return 'CHOS'
        #     elif self.N>0:
        #         return 'CHNO'
        #     else:
        #         return 'CHO'
        # elif(self.C>0 and self.H>0):
        #     return 'CHN'
        # else:
        #     return ''
                
        
    
        
