#!/usr/bin/env python
from setuptools import setup

setup(name='gym_control',   # package name
      version='0.0.1',      # package version
      description=" A collection of control theory enviroments compatible with the gym toolkit ",
      install_requires=['gym>=0.2.3', # dependencies
                        'pandas',
                        'cfg_load']
      )
