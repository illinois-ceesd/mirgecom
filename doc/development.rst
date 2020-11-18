Development How-To
==================

What Packages are Involved?
---------------------------

:mod:`mirgecom` relies on a number of other packages to do its job, which
depend on each other as illustrated in this graph:

.. graphviz::

   digraph deps {
        mirgecom -> meshmode;
        grudge -> meshmode;
        mirgecom -> grudge;

        mirgecom -> loopy;
        grudge -> loopy;
        meshmode -> loopy;

        meshmode -> pyopencl;
        loopy -> pyopencl;

        meshmode -> modepy;

        mirgecom -> pymbolic;
        loopy -> pymbolic;
   }

What do these packages do?

.. todo::

   Write this.

The source repository (and current branch) of each of these packages
in use is determined by the file
`requirements.txt in mirgecom <https://github.com/illinois-ceesd/mirgecom/blob/master/requirements.txt>`__.


Working with Pull Requests
--------------------------

We are using GitHub's pull requests (PRs) feature to integrate changes into
mirgecom and its supporting packages. Pull requests are based on git branches that are merged into another
branch (usually the master branch). Note that pull requests are a GitHub
feature and live outside the main git functionality; the `git` program itself
has no knowledge of them.

Forking the repository
^^^^^^^^^^^^^^^^^^^^^^

In most cases, it will be necessary to fork the repository (that is, create a
copy of the repository in your own GitHub account), since creating a pull
requests requires creating a branch in the repo, which requires write access.
You can fork a repository directly on the GitHub website.

Creating a new pull request
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Create a new branch for your pull request

   .. code:: bash

      $ cd mirgecom           # or loopy, meshmode, ...
      $ git branch featureX   # Create new branch for your feature
      $ git checkout featureX # Activate the branch
      # Edit files

2. Check that your changes pass tests

   .. code:: bash

      # Install these with 'pip install wheel flake8 pep8-naming flake8-quotes pytest pylint'
      $ flake8
      $ pydocstyle
      $ pytest

3. Commit your changes to the new branch

   .. code:: bash

      $ git commit

4. Push your changes

   .. code:: bash

      $ git push -u
      # alternatively:
      $ git push -u origin featureX

   If you do not have write privileges to this repository, you can push
   the change to your fork instead:

   .. code:: bash

      $ git push -u <forked_repo> featureX

5. Create pull request:

   https://github.com/illinois-ceesd/mirgecom/pulls

   The `base` branch should be the `master` branch of the repo you want to
   merge into in most cases. The `compare` branch is the branch with your
   changes.

   When creating the pull request, select at least one reviewer (someone that
   has knowledge about the code you are modifying), add yourself as the
   assignee, and choose appropriate labels (if any). Note that this can
   usually not be done for a PR from a forked repository.

6. After the pull request has been merged, you can delete the branch
   (locally and remotely):

   .. code:: bash

      $ git branch -d featureX    # delete branch locally
      $ git push –delete featureX # delete it remotely; or delete in web interface

Updating a pull request
^^^^^^^^^^^^^^^^^^^^^^^

Commit to the same local branch and push that branch:

.. code:: bash

   $ git commit
   $ git push

When changing the history of a branch (e.g., by rebasing the branch, or
by amending a commit that is already pushed), you might need to
force-push it back to the repository (i.e, ``git push --force``). Please
use this sparingly.

Reviewing/CI
^^^^^^^^^^^^

Each pull requests for mirgecom needs one manual approval by a reviewer and
needs to pass the Continuous Integration (CI) tests before merging. We use
GitHub actions as the CI provider to test each pull request. The CI tests are
triggered automatically when a pull request is created or updated.

Merging a pull request
^^^^^^^^^^^^^^^^^^^^^^

There are three ways of merging a pull request in the web interface: **squash
and merge**, **rebase and merge**, and **create a merge commit**.

Squash and merge
~~~~~~~~~~~~~~~~

Squash all commits into one commit and merge it to the main branch. This is
the preferred option, especially for small changes, as it keeps the history
shorter and cleaner, makes git bisection easier, and makes it easier to revert
a pull request.

Rebase and merge
~~~~~~~~~~~~~~~~

Rebase all commits to top of the main branch and merge all commits. This
is the preferred option for larger changes, for example, by having
separate commits for the implementation of a feature and its
documentation

Other possibilities (such as squashing only some commits and then
merging multiple commits into ``master``) are not directly supported by
GitHub’s Web UI, but can be done manually on the command line (these
might need to be force pushed to a branch).

Create a merge commit
~~~~~~~~~~~~~~~~~~~~~

This options just merges all commits into the master branch. This is the simplest
way to merge a pull request, but can lead to issues with bisection and reverting PRs
later.

Tools
^^^^^

Apart from the `git` tool, there are other tools that help to simplify various
aspects of working with GitHub:

Command line:
~~~~~~~~~~~~~

-  https://hub.github.com/
-  https://github.com/cli/cli

GUI
~~~

-  Fork
-  GitHub Desktop
-  Sublime Merge


Overview of the Setup
---------------------

The `emirge repository <https://github.com/illinois-ceesd/emirge>`__ contains some
scripts to help with installation and simultaneously has its checkout serve as a root
directory for development.

.. todo:

    - Conda environment
    - Editable installation

Installation
------------

See the installation instructions for the `emirge
<https://github.com/illinois-ceesd/emirge/>`_ installation infrastructure.

.. note::

    Should we move those here?

Installing on Your Personal Machine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    These instructions work on macOS or Linux. If you have a Windows machine, try
    `WSL <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`__.
    If that works, please submit a pull request updating this documentation
    with a procedure that worked for you.

Installing on a Cluster/DOE Machine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. todo::

   Write this.

Proposing Changes
-----------------

.. todo::

   Write this.

Building this Documentation
---------------------------

The following should do the job::

    # make sure your conda env is active
    conda install sphinx graphviz
    cd mirgecom/doc
    make html

After that, point a browser at :file:`mirgecom/doc/_build/html/index.html` to
see your documentation.
