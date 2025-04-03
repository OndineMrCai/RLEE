from setuptools import setup, find_packages

setup(
    name='rlee',
    version='0.0.0',
    description='Open-source training recipe for balance exploration and exploitation.',
    author='Agentica Team',
    packages=find_packages(include=['rlee',]),
    install_requires=[
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)