# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 15:07:44 2021

@author: Gabriel
"""

import matplotlib.pyplot as plt

plt.figure()
plt.plot([1,2], [1,2])

# Option 1
# QT backend
manager = plt.get_current_fig_manager()
manager.window.showMaximized()

# Option 2
# TkAgg backend
manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())

# Option 3
# WX backend
manager = plt.get_current_fig_manager()
manager.frame.Maximize(True)

plt.show()
plt.savefig('sampleFileName.png')

#matplotlib.get_backend()