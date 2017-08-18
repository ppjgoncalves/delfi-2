from setuptools import setup

exec(open('delfi/version.py').read())

setup(
    name='delfi',
    version=__version__,
    description='Density estimation likelihood-free inference',
    url='https://github.com/mackelab/delfi',
    install_requires=['dill', 'lasagne', 'numpy', 'scipy', 'theano', 'tqdm'],
    dependency_links = [
      'https://github.com/Lasagne/Lasagne/archive/master.zip#egg=lasagne',
    ]
)
