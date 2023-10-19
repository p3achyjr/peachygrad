import sys, os

DIR = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(DIR, "build"))

from ops import *
