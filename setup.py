from setuptools import setup, find_packages

REQUIRED_PACKAGES = ['pandas','pyarrow']

setup(
    name='trainer',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    description='My training application.'
)