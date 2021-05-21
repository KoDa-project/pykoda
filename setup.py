from setuptools import setup, find_packages

requirements = open('requirements.txt').readlines()
# Main setup:
setup(name='pykoda', version='0.2',
      description='',
      url='https://gitlab.com/savanticab/projects/vinnova/koda',
      author='David Menéndez Hurtado, Leyla Isaeva, and Samuel Lampa',
      author_email='david@savantic.se',
      license='BSD-3 clause',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      scripts=['bin/koda_getdata', 'bin/koda_getstatic'],
      install_requires=requirements,
      classifiers=['Programming Language :: Python',
                   'Development Status :: 3 - Alpha',
                   'Intended Audience :: Information Technology',
                   'Topic :: Scientific/Engineering :: GIS',
                   'License :: OSI Approved :: BSD'],
      zip_safe=False)
