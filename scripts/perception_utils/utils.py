#! /usr/bin/env python
# ----------------------------------------------------------------------------
# Authors  : Cedric BUCHE (buche@enib.fr)
# Created Date: 2022
# ---------------------------------------------------------------------------


import rospkg


def get_pkg_path():
    rp = rospkg.RosPack()
    return(rp.get_path('perception_pepper'))
    
