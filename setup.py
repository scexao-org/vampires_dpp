import os
import re
from setuptools import setup, find_packages

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
    maintainer="Miles Lucas <mdlucas@hawaii.edu>",
    license="MIT",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    package_data={},
    scripts=["scripts/vpp", "scripts/lucky_image"],
    install_requires=[
        "astropy>=4,<6",
        "numpy>=1.16,<2",
        "pandas>=1.2,<2",
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
        "docs": {"sphinx>=4.5,<5", "myst_nb==0.13", "sphinx_book_theme==0.3"},
    },
)
