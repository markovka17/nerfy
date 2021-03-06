import setuptools


def get_requirements():
    with open('requirements.txt', 'r') as f:
        requirements = f.read().splitlines()

    return requirements


setuptools.setup(
    name="nerfy",
    version="0.0.1",
    author="",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    install_requires=get_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
