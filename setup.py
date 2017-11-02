from setuptools import setup

exec(open('delfi/version.py').read())

setup(
    name='delfi',
    version=__version__,
    description='Density estimation likelihood-free inference',
    url='https://github.com/mackelab/delfi',
    author='Mackelab',
    packages=['delfi', 'delfi.distribution', 'delfi.distribution.mixture',
              'delfi.generator', 'delfi.inference', 'delfi.kernel',
              'delfi.neuralnet', 'delfi.neuralnet.layers',
              'delfi.neuralnet.loss', 'delfi.simulator', 'delfi.summarystats',
              'delfi.utils'],
    license='BSD',
    install_requires=['dill', 'lasagne==dev', 'numpy', 'scipy', 'theano', 'tqdm'],
    dependency_links=[
        'https://github.com/Lasagne/Lasagne/archive/master.zip#egg=lasagne-dev',
    ]
)
