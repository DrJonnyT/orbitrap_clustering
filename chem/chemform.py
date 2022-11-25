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
        
    
        
