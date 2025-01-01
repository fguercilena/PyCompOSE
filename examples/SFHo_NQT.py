# This example shows how to create an NQT table from an extant .h5
#
# Instructions
# - Run the 'SFHo.py' example to obtain 'SFHo.h5'
# - Run the command 'python setup.py build' in the folder
#    ../compose/NQTs to build the NQT library
# - Run this script to produce 'SFHo_NQT.h5'

import os
SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append(os.path.join(SCRIPTDIR, os.pardir))
from compose.utils import convert_to_NQTs

input_table_fname  = "SFHo.h5"
output_table_fname = "SFHo_NQT.h5"

input_table_loc  = os.path.join(SCRIPTDIR, "SFHo", input_table_fname)
output_table_loc = os.path.join(SCRIPTDIR, "SFHo", output_table_fname)

convert_to_NQTs(input_table_loc, output_table_loc, NQT_order=2, use_bithacks=True)