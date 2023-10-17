#! /usr/bin/env python
# ----------------------------------------------------------------------------
# Authors  : Cedric BUCHE (buche@enib.fr)
# Created Date: 2022
# ---------------------------------------------------------------------------


import os

def get_pkg_path():
    # get current path of this script
    current_path = os.path.dirname(os.path.realpath(__file__))
    current_path = os.path.join(current_path, '../..')
    return(current_path)
    
