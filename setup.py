# !/usr/bin/env python


def main():
    from setuptools import find_packages, setup

    version_dict = {}
    init_filename = "mirgecom/version.py"
    exec(
        compile(open(init_filename).read(), init_filename, "exec"),
        version_dict)

    setup(name="mirgecom",
          version=version_dict["VERSION_TEXT"],
          description=("TBD"),
          long_description=open("README.md").read(),
          author="CEESD",
          author_email="inform@tiker.net",
          license="MIT",
          url="https://github.com/illinois-ceesd/mirgecom",
          classifiers=[
              "Development Status :: 1 - Planning",
              "Intended Audience :: Developers",
              "Intended Audience :: Other Audience",
              "Intended Audience :: Science/Research",
              "License :: OSI Approved :: MIT License",
              "Natural Language :: English",
              "Programming Language :: Python",
              "Programming Language :: Python :: 3",
              "Topic :: Scientific/Engineering",
              "Topic :: Scientific/Engineering :: Information Analysis",
              "Topic :: Scientific/Engineering :: Mathematics",
              "Topic :: Scientific/Engineering :: Visualization",
              "Topic :: Software Development :: Libraries",
              "Topic :: Utilities",
              ],

          packages=find_packages(),

          python_requires="~=3.8",

          install_requires=[
              "mpi4py>=3",
              "pymetis",
              "pytest>=2.3",
              "pytools>=2018.5.2",
              "modepy>=2013.3",
              "arraycontext>=2021.1",
              "meshmode>=2013.3",
              "pyopencl>=2013.1",
              "pymbolic>=2013.2",
              "loopy>=2020.2",
              "cgen>=2013.1.2",
              "leap>=2019.1",
              "dagrt>=2019.1",
              "grudge>=2015.1",
              "six>=1.8",
              "logpyle",
              "importlib-resources>=1.1.0; python_version < '3.9'",
          ],

          package_data={"mirgecom": ["py.typed", "materials/*.dat"]},

          include_package_data=True,)


if __name__ == "__main__":
    main()
