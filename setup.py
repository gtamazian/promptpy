from setuptools import setup

setup(name='promptpy',
      version='0.1',
      description='Protein conformational motion simulation toolbox',
      author='Gaik Tamazian',
      packages=['promptpy', 'promptpy.commands'],
      entry_points={
          'console_scripts': ['promptpy=promptpy.__main__:cli']
      },
      install_requires=[
          'numpy',
          'h5py',
          'scipy',
          'biopython',
          'tqdm',
          'matplotlib'
      ],
      zip_safe=False)
