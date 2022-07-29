from setuptools import setup, find_packages

setup(
    name = "GlottisNetV2",
    version = "0",
    author = "Andreas Kist, Elina Kruse",
    author_email = "andreas.kist@fau.de, elinakruse@web.de",
    license = "GPLv3",
    packages=find_packages(),
    install_requires = ["tensorflow-gpu",
        "tensorflow_addons",
        "albumentations",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "scikit-image",
        "opencv-python",
        "notebook",
        "tqdm",
        "ipywidgets"],

    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3.8",
    ],
    keywords = "Glottal Midline, Anterior point, Posterior point, Segmentation",
    description="A tool to detect the glottal midline.",
    include_package_data = True,
)