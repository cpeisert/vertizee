===============================
Install
===============================

Python version support
======================

Vertizee is tested on Python 3.6, 3.7, and 3.8.


Environment
===========

You should install Vertizee in a `virtual environment <https://docs.python.org/3/library/venv.html>`_
or a `conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.
If you are unfamiliar with Python virtual environments, check out the
`user guide <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/>`_.

First, make sure you have the latest version of ``pip`` (the Python package manager)
installed. If you do not, refer to the `Pip documentation
<https://pip.pypa.io/en/stable/installing/>`_ and install ``pip`` first.

**Optional**: Consider setting up the Python scientific computing stack
(`SciPy <https://www.scipy.org/>`_: `IPython <https://ipython.org/>`_,
`NumPy <https://www.numpy.org/>`_, `Matplotlib <https://matplotlib.org/>`_, ...).
The simplest way to install it, is to use `Anaconda <https://docs.continuum.io/anaconda/>`_,
a cross-platform (Linux, Mac OS X, Windows) Python distribution for data analytics and scientific
computing.


Install the released version
============================

With Pip
--------

Install the current release of ``vertizee`` with ``pip``::

    $ pip install vertizee

To upgrade to a newer release use the ``--upgrade`` flag::

    $ pip install --upgrade vertizee

If you do not have permission to install software system-wide, you can
install into your user directory using the ``--user`` flag::

    $ pip install --user vertizee


With Conda
----------

Install the current release of Vertizee with ``conda``::

    $ conda install --channel conda-forge vertizee

To upgrade to a newer release use the `update <https://docs.conda.io/projects/conda/en/latest/commands/update.html>`_
command::

    $ conda update vertizee

**Note**: By default, conda and all packages it installs, including Anaconda, are installed locally
with a user-specific configuration. Administrative privileges are not required, and no upstream
files or other users are affected by the installation.


Install by manually downloading
===============================

Alternatively, you can manually download ``vertizee`` from
`GitHub <https://github.com/cpeisert/vertizee/releases>`_  or
`PyPI <https://pypi.python.org/pypi/vertizee>`_.
To install one of these versions, unpack it and run the following from the
top-level source directory::

    $ pip install .

Install source with Conda
-------------------------

.. code-block:: console

    $ conda activate <YOUR ENVIRONMENT>
    $ conda install pip
    $ pip install .


Install the development version
===============================

If you have `Git <https://git-scm.com/>`_ installed on your system, it is also
possible to install the development version of ``vertizee``.

Before installing the development version, you may need to uninstall the
standard version of ``vertizee``.

Install with Pip
----------------

.. code-block:: console

    $ pip uninstall vertizee
    $ git clone https://github.com/cpeisert/vertizee.git
    $ cd vertizee
    $ pip install --editable .

Install with Conda
------------------

.. code-block:: console

    $ conda remove vertizee
    $ git clone https://github.com/cpeisert/vertizee.git
    $ cd vertizee
    $ conda develop .

The ``pip install --editable .`` and ``conda develop .`` commands allow you to follow the
development branch as it changes by creating links in the right places and installing the command
line scripts.

Then, if you want to update ``vertizee`` at any time, in the same directory do::

    $ git pull


Test a source distribution
==========================

Vertizee uses the `Pytest <https://pytest.org>`_ testing package. You can test the complete package
from the unpacked source directory with::

    $ pytest vertizee
