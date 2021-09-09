import setuptools
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'robo_gym'))
from version import VERSION

setuptools.setup(name='robo-gym',
      version=VERSION,
      description='robo-gym: an open source toolkit for Distributed Deep Reinforcement Learning on real and simulated robots.',
      url='https://github.com/jr-robotics/robo-gym',
      author="Matteo Lucchi, Friedemann Zindler",
      author_email="matteo.lucchi@joanneum.at, friedemann.zindler@joanneum.at",
      packages=setuptools.find_packages(),
      include_package_data=True,
      classifiers=[
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
      ],
      install_requires=[
      'gym',
      'robo-gym-server-modules',
      'numpy',
      'pyyaml'
      ],
      python_requires='>=3.6',
      scripts = ['bin/run-rs-side-standard']
)
