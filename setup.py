from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    'numpy',
    'pandas',
    'scipy',
]
setup(
   name='dacota',
    version='0.1.1',
    description='Datategy Cohort Targeting : One-click discovery of diverse cohorts in your dataset with statistical guarantees',
    py_modules=["dacota"],
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",

        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Dr. Eren Unlu',
    author_email='eren.unlu@datategy.net',
    keywords=['Cohort', 'Cohortization', 'Data Analysis', 'Python 3', 'EDA'],
    url="FILL",
    license='MIT',
    install_requires=install_requires,
)