# -*- coding: utf-8 -*-

from pytest import approx

from chem.chemform import ChemForm

def test_ChemForm():
    chem1 = ChemForm('CH2O3S')
    assert chem1.C == 1
    assert chem1.H == 2
    assert chem1.O == 3
    assert chem1.S == 1
    
    chem2 = ChemForm('C H O2')
    assert chem2.H == 1
    assert chem2.O == 2
    assert chem2.S == 0
    
    assert chem1.classify() == 'CHOS'
    assert chem2.classify() == 'CHO'
    assert ChemForm('CHO').classify() == 'CHO'
    assert ChemForm('HOCS').classify() == 'CHOS'
    assert ChemForm('S8C15H35').classify() == 'CHS'
    assert ChemForm('N8C15H35').classify() == 'CHN'
    assert ChemForm('N8C15H35O S').classify() == 'CHONS'
    
    #Test ratios        
    assert chem1.ratios() == (2,0,3,1)
    assert ChemForm('C3 H5 N O3 S2').ratios() == approx((5/3,1/3,1,2/3))
    
    
    

