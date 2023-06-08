from setuptools import setup
from setuptools import find_packages


with open('req_small.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]


setup(name='decode',
      description = "Decoding emotion of people's faces",
      packages=find_packages(),
      install_requires=requirements)
