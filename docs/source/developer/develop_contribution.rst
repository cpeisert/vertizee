.. _contributor_development_guide:

============================================================================
Contributor Development Guide
============================================================================

Before diving into contribution development, be sure to review the :doc:`Contributor Introduction <_contributor_intro>`.

 - `Development Workflow`_
 - `Submission Guidelines`_
 - `Coding Rules`_
 - `Commit Message Format`_


Development Workflow
================================

1. **If you are a first-time contributor:**

 * Go to `https://github.com/cpeisert/vertizee <https://github.com/cpeisert/vertizee>`_ and click the
   "fork" button to create your own copy of the project.

 * Clone the project to your local computer::

    git clone git@github.com:your-username/vertizee.git

 * Navigate to the folder vertizee and add the upstream repository::

    git remote add upstream git@github.com:cpeisert/vertizee.git

 * Now, you have remote repositories named:

   - ``upstream``, which refers to the ``vertizee`` repository
   - ``origin``, which refers to your personal fork

 * Next, you need to set up your build environment.
   Here are instructions for two popular environment managers: **Pip** and **Anaconda**.


 * **Pip Build Environment: ``venv``**

   * Create a virtualenv named ``vertizee-dev`` that lives in the directory of the same name::

      python -m venv vertizee-dev

   * Activate it::

      source vertizee-dev/bin/activate

   * Install main development and runtime dependencies of vertizee::

      pip install -r <(cat requirements/{default,doc,test}.txt)

   * Build and install vertizee from source::

      pip install --editable .

   * Test your installation::

      PYTHONPATH=. pytest vertizee


 * **Anaconda Build Environment: ``conda``**

   * Create a conda named ``vertizee-dev``::

      conda create --name vertizee-dev

   * Activate it::

      conda activate vertizee-dev

   * Install main development and runtime dependencies of vertizee::

       conda install -c conda-forge `for i in requirements/{default,doc,test}.txt; do echo -n " --file $i "; done`

   * Build and install vertizee from source::

      pip install --editable . --no-deps

   * Test your installation::

      PYTHONPATH=. pytest vertizee


 * Finally, we recommend you use a pre-commit hook, which runs black when you type ``git commit``::

    pre-commit install

2. **Develop your contribution, including appropriate test cases:**

   * Pull the latest changes from upstream::

      git checkout master
      git pull upstream master

   * Create a branch for the feature you want to work on. Since the
     branch name will appear in the merge message, use a sensible name
     such as 'bugfix-for-issue-1480'::

      git checkout -b bugfix-for-issue-1480 master

   * Commit locally as you progress (``git add`` and ``git commit``)

   * Follow our :ref:`Coding Rules <section-coding-rules>`.

3. **Test your contribution:**

   * Run the test suite locally (see `Testing`_ for details)::

      PYTHONPATH=. pytest networkx

   * Running the tests locally *before* submitting a pull request helps catch
     problems early and reduces the load on the continuous integration
     system.


Submission Guidelines
================================

Submitting a Pull Request (PR)
--------------------------------

Before you submit your Pull Request (PR) consider the following guidelines:

1. Search `GitHub <https://github.com/cpeisert/vertizee/pulls>`_ for an open or closed PR that relates to your submission.
   You don't want to duplicate existing efforts.

2. Be sure that an issue describes the problem you're fixing, or documents the design for the feature you'd like to add.
   Discussing the design upfront helps to ensure that we're ready to accept your work.

3. Fork the `cpeisert/vertizee <https://github.com/cpeisert/vertizee>`_ repository.

4. Make your changes in a new git branch::

    git checkout -b my-fix-branch master

5. Create your patch, **including appropriate test cases**.

6. Follow our :ref:`Coding Rules <section-coding-rules>`.

7. Run the full Vertizee test suite, as described in the [developer documentation][dev-doc], and ensure that all tests pass.

8. Commit your changes using a descriptive commit message that follows our [commit message conventions](#commit).
   Adherence to these conventions is necessary because release notes are automatically generated from these messages.

  ::

    git commit -a

Note: the optional commit ``-a`` command line option will automatically "add" and "rm" edited files.

9. Push your branch to GitHub::

    git push origin my-fix-branch

10. In GitHub, send a pull request to ``vertizee:master``.

   If we ask for changes via code reviews then:

   * Make the required updates.
   * Re-run the Angular test suites to ensure tests are still passing.
   * Rebase your branch and force push to your GitHub repository (this will update your Pull Request):

    ::

      git rebase master -i
      git push -f

That's it! Thank you for your contribution!


After your pull request is merged
---------------------------------

After your pull request is merged, you can safely delete your branch and pull the changes from the main (upstream) repository:

* Delete the remote branch on GitHub either through the GitHub web UI or your local shell as follows::

    git push origin --delete my-fix-branch

* Check out the master branch::

    git checkout master -f

* Delete the local branch::

    git branch -D my-fix-branch

* Update your master with the latest upstream version::

    git pull --ff upstream master


.. _section-coding-rules:

Coding Rules
================================
To ensure consistency throughout the source code, keep these rules in mind as you are working:

* All features or bug fixes **must be tested** by one or more specs (unit-tests).
* All public API methods **must be documented**.
* We follow `Google's Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_, but wrap all code at **100 characters**.

  * **Visual Studio Code users**: The extension *Python Docstring Generator* can be configured with the
    *docstring_template.mustache* template in the Vertizee repo. Update the Workspace setting
    **Auto Docstring: Custom Template Path** to point to ``./docstring_template.mustache``


Commit Message Format
================================

The following Git commit message formatting rules lead to **easier to read commit history**.

Each commit message consists of a **header**, a **body**, and a **footer**::

    <header>
    <BLANK LINE>
    <body>
    <BLANK LINE>
    <footer>

The **header** is mandatory and must conform to the `Commit Message Header`_ format.

The **body** is mandatory for all commits except for those of scope "docs".
When the body is required it must be at least 20 characters long.

The **footer** is optional.

Any line of the commit message cannot be longer than 100 characters.


Commit Message Header
---------------------

::

    <type>(<scope>): <short summary>
    │       │             │
    │       │             └─⫸ summary in present tense; not capitalized; no period at the end
    │       │
    │       └─⫸ Commit Scope: classes|algorithms|io|changelog|dev-infra
    │
    └─⫸ Commit Type: build|ci|docs|feat|fix|perf|refactor|style|test


The ``<type>`` and ``<summary>`` fields are mandatory, the ``(<scope>)`` field is optional.


Type
----------------

Must be one of the following:

* **docs**: Documentation only changes
* **feat**: A new feature
* **fix**: A bug fix
* **perf**: A code change that improves performance
* **refactor**: A code change that neither fixes a bug nor adds a feature
* **test**: Adding missing tests or correcting existing tests


Scope
----------------

The scope should be the name of the package affected.

The following is the list of supported scopes:

* ``classes``
* ``algorithms``
* ``io``

There are currently a few exceptions to the "use package name" rule:

* ``changelog``: used for updating the release notes in CHANGELOG.rst

* ``dev-infra``: used for development infrastructure related changes such as updating pylintrc or setup.py

* none/empty string: useful for ``style``, ``test`` and ``refactor`` changes that are done across all packages and for docs changes that are not related to a specific package (e.g. ``docs: fix typo in tutorial``)


Summary
----------------

Use the summary field to provide a succinct description of the change:

* use the imperative, present tense: "change" not "changed" nor "changes"
* don't capitalize the first letter
* no dot (.) at the end


Commit Message Body
--------------------------------

Just as in the summary, use the imperative, present tense: "fix" not "fixed" nor "fixes".

Explain the motivation for the change in the commit message body. This commit message should explain _why_ you are making the change.
You can include a comparison of the previous behavior with the new behavior in order to illustrate the impact of the change.


Commit Message Footer
--------------------------------

The footer can contain information about breaking changes and is also the place to reference GitHub issues and other PRs that this commit closes or is related to.

```
BREAKING CHANGE: <breaking change summary>
<BLANK LINE>
<breaking change description + migration instructions>
<BLANK LINE>
<BLANK LINE>
Fixes #<issue number>
PR Close #<issue number>
```

Breaking Change section should start with the phrase "BREAKING CHANGE: " followed by a summary of the breaking change, a blank line, and a detailed description of the breaking change that also includes migration instructions.


Revert commits
--------------------------------

If the commit reverts a previous commit, it should begin with `revert: `, followed by the header of the reverted commit.

The content of the commit message body should contain:

- information about the SHA of the commit being reverted in the following format: `This reverts commit <SHA>`,
- a clear description of the reason for reverting the commit message.


Testing
-------

Vertizee has an extensive test suite that ensures correct execution on your system.
The test suite has to pass before a pull request can be merged, and tests should be added to cover any
modifications to the code base.
We make use of the `pytest <https://docs.pytest.org/en/latest/>`__ testing framework, with tests located in the various
``vertizee/submodule/tests`` folders.

To run all tests::

    $ PYTHONPATH=. pytest vertizee

Or the tests for a specific submodule::

    $ PYTHONPATH=. pytest vertizee/readwrite

Or tests from a specific file::

    $ PYTHONPATH=. pytest vertizee/readwrite/tests/test_yaml.py

Or a single test within that file::

    $ PYTHONPATH=. pytest vertizee/readwrite/tests/test_yaml.py::TestYaml::testUndirected

Use ``--doctest-modules`` to run doctests.
For example, run all tests and all doctests using::

    $ PYTHONPATH=. pytest --doctest-modules vertizee

Tests for a module should ideally cover all code in that module,
i.e., statement coverage should be at 100%.

To measure the test coverage, run::

  $ PYTHONPATH=. pytest --cov=vertizee vertizee

This will print a report with one line for each file in Vertizee,
detailing the test coverage::

  Name                                             Stmts   Miss Branch BrPart  Cover
  ----------------------------------------------------------------------------------
  vertizee/__init__.py                                33      2      2      1    91%
  vertizee/algorithms/__init__.py                    114      0      0      0   100%
  vertizee/algorithms/approximation/__init__.py       12      0      0      0   100%
  vertizee/algorithms/approximation/clique.py         42      1     18      1    97%
  ...

