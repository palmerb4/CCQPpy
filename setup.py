import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='ccqppy',
    author='Bryce Palmer',
    author_email='palmerb4@wit.edu',
    description='A systematic comparison of various algorithms for solving convex constrained quadratic programming problems',
    keywords='quadratic programming, convex optimization',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/palmerb4/ccqppy',
    project_urls={
        'Documentation': 'https://github.com/palmerb4/ccqppy',
        'Bug Reports':
        'https://github.com/palmerb4/ccqppy/issues',
        'Source Code': 'https://github.com/palmerb4/ccqppy',
        # 'Funding': '',
        # 'Say Thanks!': '',
    },
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    classifiers=[
        # see https://pypi.org/classifiers/
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=['matplotlib', 'numpy', 'scipy'],
    extras_require={
        'dev': ['check-manifest'],
        # 'test': ['coverage'],
    },
    # entry_points={
    #     'console_scripts': [  # This can provide executable scripts
    #         'run=ccqppy:main',
    # You can execute `run` in bash to run `main()` in src/ccqppy/__init__.py
    #     ],
    # },
)
