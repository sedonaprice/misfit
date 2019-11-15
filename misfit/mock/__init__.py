# Copyright 2018 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE.rst

try:
    from generate_basic import generate_mock_IFU_cube, generate_mock_slit_obs
except:
    from .generate_basic import generate_mock_IFU_cube, generate_mock_slit_obs
