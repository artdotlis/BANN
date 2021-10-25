# -*- coding: utf-8 -*-
# https://packaging.python.org/tutorials/packaging-projects/
from pathlib import Path

import setuptools  # type: ignore
from bann.version import BANN_VERSION

with Path("README.md").open("r") as fh_read_me:
    READ_ME = fh_read_me.read()

setuptools.setup(
    name="BANN",
    version=BANN_VERSION,
    author="Artur Lissin",
    author_email="arturOnRails@protonmail.com",
    description="Library for building artificial neural networks.",
    long_description=READ_ME,
    long_description_content_type="text/markdown",
    url="",
    license='MIT License',
    packages=setuptools.find_packages(),
    python_requires="==3.8",
    install_requires=[
        'ReWoWr == 0.8.0',
        'PAN == 0.8.1',
        'seaborn == 0.9.0',
        'matplotlib == 3.1.2',
        'pandas == 0.25.3',
        'numpy == 1.18.1',
        'torch == 1.4.0',
        'torchvision == 0.5.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License"
    ],
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'bann=bann.main:main',
            'bann_info=bann.info_spec:info_spec',
            'bann_info_gen=bann.info_gen:info_gen',
            'bann_lnss=bann.load_net_save_state:load_net_save_state',
            'bann_lnso=bann.load_net_save_onnx:load_net_save_onnx'
        ]
    }
)
