.. deephyp documentation master file, created by
   sphinx-quickstart on Thu Aug 29 19:50:37 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation
=============

The `latest release
<https://pypi.org/project/deephyp/>`_ of the toolbox can be installed from the command line using pip:

.. code-block:: bash

   pip install deephyp

or to update:

.. code-block:: bash

   pip install deephyp --upgrade

The software dependencies needed to run the toolbox are python 2 or python 3 (tested with version 2.7.15 and 3.5.2) \
with packages:

   - tensorflow (tested with v1.14.0) - not yet compatible with tensorflow v2.0
   - numpy (tested with v1.15.4)

Because deephyp is note yet compatible with tensorflow v2.0, you will have to install an older version of tensorflow:

.. code-block:: bash

   pip install tensorflow==1.14

Or if you are using a gpu:

.. code-block:: bash

   pip install tensorflow-gpu==1.14

If you want to use deephyp but you have tensorflow v2.0 installed, you can install deephyp in a virtual environment with \
tensorflow v1.14. `See instructions on setting up a virtual environment here
<https://www.tensorflow.org/install/pip>`_.

To import deephyp, in python write:

.. code-block:: python

   import deephyp

Source code available on `github
<https://github.com/lloydwindrim/hyperspectral-autoencoders>`_.

