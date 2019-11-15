# misfit/tests/test_model2d.py
# Test functionality, speed of 2D MISFIT models
# 
# Copyright 2019 Sedona Price <sedona.price@gmail.com>.
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from unittest import TestCase

import misfit

class TestModel2D(TestCase):
    def test_is_string(self):
        s = funniest.joke()
        self.assertTrue(isinstance(s, basestring))