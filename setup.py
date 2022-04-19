from setuptools import setup, find_packages
import os
import re

version = ""

with open(os.path.join("vampires_dpp", "__init__.py"), "r") as fh:
    for line in fh.readlines():
        m = re.search("__version__ = [\"'](.+)[\"']", line)
        if m:
            version = m.group(1)


with open("README.md", "r") as fh:
    readme = fh.read()

setup(
    long_description=readme,
    long_description_content_type="text/markdown",
    name="vampires_dpp",
    version=version,
    description="Tools for processing VAMPIRES data",
    python_requires=">=3.7,<3.10",
    project_urls={
        "repository": "https://github.com/scexao-org/vampires_dpp",
    },
    author="Miles Lucas",
    author_email="mdlucas@hawaii.edu",
    maintainer="Miles Lucas <mdlucas@hawaii.edu",
    license="MIT",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    package_data={},
    scripts=["scripts/vpp", "scripts/lucky_image"],
    install_requires=[
        "astropy>=4,<6",
        "numpy>=1.16,<2",
        "scikit-image>=0.18,<0.20",
        "scipy>=1.7,<2",
        "tqdm==4.*",
    ],
    extras_require={
        "test": [
            "pytest==7.*",
            "pytest-cov==3.*",
            "pytest-randomly==3.*",
            "black==22.*",
            "pytest-black==0.3.*",
        ],
        "docs": {
            "mkdocs>=1.3,<2",
            "mkdocstrings[python]==0.18.*",
            "mkdocs-autorefs==0.4.*",
        },
    },
)
