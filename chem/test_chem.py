# -*- coding: utf-8 -*-

from chem.chemform import chemform

def test_chemform():
    chem1 = chemform('CH2O3S')
    assert chem1.C == 1
    assert chem1.H == 2
    assert chem1.O == 3
    assert chem1.S == 1
    
    chem2 = chemform('C H O2')
    assert chem2.H == 1
    assert chem2.O == 2
    assert chem2.S == 0
    
    assert chem1.classify() == 'CHOS'
    assert chem2.classify() == 'CHO'
    assert chemform('CHO').classify() == 'CHO'
    assert chemform('HOCS').classify() == 'CHOS'
    assert chemform('S8C15H35').classify() == 'CHS'
    assert chemform('N8C15H35').classify() == 'CHN'
    assert chemform('N8C15H35O S').classify() == 'CHONS'
    
    
    

