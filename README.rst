==========
pyfw-lasso
==========

This python package provides two Frank-Wolfe algorithms to solve the LASSO problem
   
.. image:: /doc/img/lasso.png
  :width: 25 %
  :align: center

These algorithms are:

* the Vanilla Frank-Wolfe algorithm,
* the Polyatomic Frank-Wolfe algorithm.

Polyatomic Frank-Wolfe
======================

The **Polyatomic Frank-Wolfe** algorithm has been proposed and described there: https://doi.org/10.1109/LSP.2022.3149377 (pre-print version available).

It is an optimizatoin algorithm that builds apon the classical Frank-Wolfe algorithm by allowing to place *multiple atoms* at each iteration. This results is a significantly faster convergence. An additional *approximate correction step* is used in order to accelerate further more the solving time while preserving the *accuracy of the solution*.

Installation
===========

Instructions are provided in the `howtoinstall.txt` file.


.. code-block:: bash
   
   $ git clone git@github.com:AdriaJ/pyfw-lasso.git
   $ cd pyfw-lasso
   $ conda create --name pyfwl --strict-channel-priority --channel=conda-forge --file=conda/requirements.txt
   $ conda activate pyfwl
   $ pip install -e .
   $ pytest

Citation
========

For citing this package, please refer to the Signal Processing Letters article: https://doi.org/10.1109/LSP.2022.3149377 .
