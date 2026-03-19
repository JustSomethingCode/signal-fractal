from setuptools import setup, find_packages

setup(
    name='fib_mask',
    version='1.0.0',
    description='O(1) Sparse Fibonacci Scatter-Shield Masks for PyTorch',
    author='AI Research',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy'
    ],
)
