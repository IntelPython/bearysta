from setuptools import setup, find_packages
setup(
    name='bearysta',
    version='0.1.2',
    packages=find_packages(),
    install_requires=["pandas", "ruamel_yaml", "openpyxl", "matplotlib"],
    package_data={
        "bearysta": ['html/*.html']
    }
)
