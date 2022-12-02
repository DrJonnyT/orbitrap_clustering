# -*- coding: utf-8 -*-
from plotting.cmap_EOS11 import cmap_EOS11

from matplotlib.colors import LinearSegmentedColormap

def test_cmap_EOS11():
    cmap = cmap_EOS11()
    assert type(cmap) == LinearSegmentedColormap
    assert cmap.N == 11
    assert cmap_EOS11(50).N == 50
    
