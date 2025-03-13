from setuptools import find_packages, setup

setup(
    name="robot_eval_logger",
    packages=find_packages(),
    version="0.0.1",
    install_requires=[
        "absl-py",
        "wandb",
        "moviepy==1.0.3",
        "huggingface-hub",
        "numpy<2.0.0",
    ],
)
