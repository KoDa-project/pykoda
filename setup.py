from setuptools import setup, find_packages

requirements = open('requirements.txt').readlines()
# Main setup:
setup(name='pykoda', version='0.2',
      description='',
      url='https://gitlab.com/savanticab/projects/vinnova/koda',
      author='David Menéndez Hurtado and Leyla Isaeva',
      author_email='david@savantic.se',
      license='',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      scripts=['bin/koda_getdata', 'bin/koda_getstatic'],
      install_requires=requirements,
      classifiers=['Programming Language :: Python',
                   'Development Status :: 3 - Alpha',
                   'Intended Audience :: Information Technology',
                   'Topic :: Scientific/Engineering :: GIS'],
      zip_safe=False)
