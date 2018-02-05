from setuptools import setup, find_packages

setup(
    name='HyperparameterLogger',
    version='0.2.dev0',
    description='Simple model tracker, that wraps Keras, Sklearn and Gensim to record hyperparameters.',
    author='Umut Karakulak',
    author_email='umut@dunksoft.com',
    url='https://www.dunksoft.com',
    license='MIT',
    packages=find_packages(),
    install_requires=['PyYAML', 'xmltodict']
)