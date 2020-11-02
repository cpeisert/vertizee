#!/usr/bin/env python
# Copyright 2020 The Vertizee Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Setup script for Vertizee.

You can install Vertizee with:
python setup.py install
"""
#pylint: disable=exec-used, undefined-variable

import sys
from setuptools import find_packages, setup

DISTNAME = 'vertizee'
DESCRIPTION = 'An object-oriented, typed, graph library for the analysis and study of graphs.'
with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'
AUTHOR = f'The {DISTNAME.title()} Authors'
AUTHOR_EMAIL = 'cpeisert@gmail.com'
MAIN_PACKAGE = 'vertizee'
LICENSE = 'Apache 2.0'
ORG_OR_USER = 'cpeisert'
PYTHON_REQUIRES = '>=3.7'
REQUIREMENTS = [
    requirement.strip() for requirement in open('requirements/default.txt').readlines()
]

DOCS_REQUIREMENTS = [
    requirement.strip() for requirement in open('requirements/docs.txt').readlines()
]
TEST_REQUIREMENTS = [
    requirement.strip() for requirement in open('requirements/test.txt').readlines()
]
EXTRA_REQUIREMENTS = {
    'all': DOCS_REQUIREMENTS + TEST_REQUIREMENTS
}
URL = f'https://github.io/{ORG_OR_USER}/{DISTNAME}'


if sys.argv[-1] == "setup.py":
    print('To install, run "python setup.py install"')
    print()

if sys.version_info[:2] < (3, 6):
    ver: tuple = sys.version_info[:2]
    error = f'{DISTNAME.title()} requires Python 3.6 or later ({ver[0]}.{ver[1]} detected).'
    sys.stderr.write(error + "\n")
    sys.exit(1)

# Read the __version__ variable.
with open(f'{DISTNAME}/version.py') as f:
    exec(f.read(), globals())

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version=__version__,
        url=URL,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        python_requires=PYTHON_REQUIRES,
        install_requires=REQUIREMENTS,
        extras_require=EXTRA_REQUIREMENTS,
        license=LICENSE,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
        package_data={MAIN_PACKAGE: ["py.typed"]},
        packages=find_packages(),
    )
