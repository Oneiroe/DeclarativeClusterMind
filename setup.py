from setuptools import setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='DeclarativeClusterMind',
    version='1.0',
    packages=['DeclarativeClusterMind'],
    url='https://github.com/Oneiroe/ClusterMind',
    license=license,
    author='Alessio Cecconi',
    author_email='alessio.cecconi.1991@gmail.com',
    description='Trace Clustering based on Declarative Logic Rules',
    long_description=readme,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU GPL3',

        # Specify the Python versions you support here. In particular, ensure
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='trace clustering declare declarative rules temporal logic decision trees',
)
