import os
from setuptools import setup


def readme():
    """Import the README.md Markdown file and try to convert it to RST format."""
    try:
        import pypandoc

        return pypandoc.convert_file('README.md', 'rst')
    except(IOError, ImportError):
        with open('README.md') as readme_file:
            return readme_file.read()


setup(
    name='disaster_tweet_detect',
    version='0.1',
    description='NLP with disaster tweets',
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    url='https://github.com/matthewkhung/nlp_capstone',
    author='Matthew Hung',
    author_email='matthew.k.hung@gmail.com',
    license='MIT',
    packages=['disaster_tweet_detect'],
    install_requires=[
        'pypandoc>=1.11',
        'watermark>=2.3.1',
        'pandas>=1.5.3',
        'matplotlib<=3.7.0',
        'nltk<=3.8.1',
        'numpy<=1.24.2',
        'scikit-learn>=1.2.1',
        'scipy<=1.10.0',
        'seaborn<=0.12.2',
        'statsmodels<=0.13.5',
    ]
)
