Building this Documentation
===========================

The following should do the job::

    # make sure your conda env is active
    conda install sphinx graphviz
    pip install furo sphinx-copybutton sphinx-math-dollar
    cd mirgecom/doc
    make html

After that, point a browser at :file:`mirgecom/doc/_build/html/index.html` to
see your documentation.
