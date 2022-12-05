# -*- coding: utf-8 -*-
import numpy as np
from clustering import molecule_type_pos_frac
from clustering.molecule_type_pos_frac import molecule_type_pos_frac_clusters


def test_molecule_type_pos_frac():
    mol_types = ['CHO','CHO','CHON','CHON','CHONS','CHOS']
    data = np.array([1,1,-1,1,0,-1])
    mols_list = np.unique(mol_types)
    assert np.array_equal( molecule_type_pos_frac(data,mol_types,mols_list=mols_list).to_numpy(), [1,0.5,1,0])
    
    
def test_molecule_type_pos_frac_clusters():
    mol_types = ['CHO','CHO','CHON','CHON','CHONS','CHOS']
    data = np.array([[1,1,-1,1,0,-1],[-1,-1,-1,1,0,-1]])
    mols_list = np.unique(mol_types)
    clusters = [0,1]
    
    assert np.array_equal( molecule_type_pos_frac_clusters(data,mol_types,clusters,mols_list=mols_list).to_numpy(), 
                          np.array([[1,0.5,1,0],
                          [0,0.5,1,0]]))
    