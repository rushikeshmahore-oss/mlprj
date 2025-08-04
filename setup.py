from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path):
    with open(file_path) as f:
        lines = f.read().splitlines()
        return [line for line in lines if line and not line.startswith(("-e", "#"))]


        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='mlprj',
    version='0.0.1',
    author='Vaibhavi',
    author_email='vaibhavizaskar20@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)