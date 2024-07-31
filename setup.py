from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Fourier ptychography optics module for python'
LONG_DESCRIPTION = 'Allows for the computation of high resolution images from low resolution images in differing k-space using Fourier ptychography'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="fpm_opt", 
        version=VERSION,
        author="Robin Cunningham",
        author_email="<rspcunningham@gmail.com>",
        maintainer="Sammy Farnum",
        maintainer_email="<sfarnum1132@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independant",
        ]
)