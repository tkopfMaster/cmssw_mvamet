import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.mlab as mlab
import ROOT
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import scipy.stats
import matplotlib.ticker as mtick
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter, MaxNLocator
from getPlotsOutputclean import loadData, loadData_woutGBRT
import h5py
import sys
from matplotlib.lines import Line2D
import numpy as np
import root_numpy as rnp
from scipy import optimize
import ROOT
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import h5py
import sys
import time
from scipy.stats import chisquare
