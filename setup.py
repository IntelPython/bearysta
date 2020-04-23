from setuptools import setup, find_packages
setup(
    name='bearysta',
    version='0.1',
    packages=find_packages(),
    install_requires=["pandas", "ruamel_yaml"]
)
