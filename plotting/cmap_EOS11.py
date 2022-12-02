# -*- coding: utf-8 -*-
from matplotlib.colors import LinearSegmentedColormap
def cmap_EOS11():
    """
    
    Returns
    -------
    cmap : matplotlib cmap
        EOS11 cmap from Igor Pro

    """
    colors = [(157/255, 30/255, 55/255),(205/255,58/255,70/255),(233/255,111/255,103/255),(242/255,162/255,121/255),(247/255,209/255,152/255),(242/255,235/255,185/255),(207/255,231/255,239/255),(138/255,209/255,235/255),(58/255,187/255,236/255),(0,154/255,219/255), (0, 94/255, 173/255)]
    cmap = LinearSegmentedColormap.from_list("EOSSpectral11", colors,N=11)
    return cmap