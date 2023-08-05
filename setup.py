from setuptools import setup
from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="modeldiff",
    version="0.0.1",
    packages=find_packages(),
    license='MIT License',
    description='Repo for XADD model-diff experiments',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Xiaotian Zhu and Jihwan Jeong',
    # author_email='',
    url='https://github.com/jackliuto/model_diff_XADD',
    # download_url='',
)