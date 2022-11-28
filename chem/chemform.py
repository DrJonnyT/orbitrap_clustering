# -*- coding: utf-8 -*-
import re
from numpy import NaN

class ChemForm:
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
        
    def ratios(self):
        if(self.C == 0):
            return NaN, NaN, NaN, NaN
        else:
            H_C = self.H / self.C
            O_C = self.O / self.C
            N_C = self.N / self.C
            S_C = self.S / self.C
            return H_C, N_C, O_C, S_C
                
        
    
        
