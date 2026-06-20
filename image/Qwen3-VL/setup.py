"""
Setup script for PriorTR on Qwen3-VL.

Usage:
    python setup.py develop

This will:
    1. Register `visual_token_pruning` as an importable package
    2. Create a symlink from transformers/models/qwen3_vl -> ./qwen3/
       so that the custom model implementation (with VTR hooks) is used
"""

import os
import sys

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


def _get_transformers_qwen3_vl_path():
    """Find the transformers/models/qwen3_vl directory."""
    try:
        import transformers
        return os.path.join(os.path.dirname(transformers.__file__), "models", "qwen3_vl")
    except ImportError:
        return None


def _create_symlink():
    """Create symlink: transformers/models/qwen3_vl -> ./qwen3/"""
    target = _get_transformers_qwen3_vl_path()
    if target is None:
        print("[PriorTR] Warning: transformers not installed. "
              "Install transformers first, then re-run: pip install -e .")
        return False

    source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3")

    # Already a symlink pointing to the right place
    if os.path.islink(target) and os.path.realpath(target) == os.path.realpath(source):
        print(f"[PriorTR] Symlink already exists: {target} -> {source}")
        return True

    # Backup original directory
    backup = target + "_backup"
    if os.path.isdir(target) and not os.path.islink(target):
        if not os.path.exists(backup):
            os.rename(target, backup)
            print(f"[PriorTR] Backed up original: {target} -> {backup}")
        else:
            import shutil
            shutil.rmtree(target)
            print(f"[PriorTR] Removed existing directory (backup already exists): {target}")
    elif os.path.islink(target):
        os.remove(target)
        print(f"[PriorTR] Removed existing symlink: {target}")

    os.symlink(source, target)
    print(f"[PriorTR] Created symlink: {target} -> {source}")
    return True


def _run_then_symlink(super_run):
    """Run the base command, then ALWAYS create the symlink.

    setuptools' legacy dependency-resolution step can abort (e.g. a transitive
    `typer` version conflict pulled in by huggingface-hub) *after* the package
    itself is already installed. That must not stop the qwen3_vl symlink — the
    symlink is what makes the custom VTR model code take effect — so we create
    it in a `finally`, surfacing any dependency-step error as a warning.
    """
    try:
        super_run()
    except BaseException as e:  # noqa: BLE001 - keep going so the symlink is made
        print(f"[PriorTR] Warning: dependency step reported: {e!r}. "
              f"The package is installed; continuing to create the symlink.")
    finally:
        _create_symlink()


class PostDevelopCommand(develop):
    """Post-installation for development mode (python setup.py develop)."""
    def run(self):
        _run_then_symlink(lambda: develop.run(self))


class PostInstallCommand(install):
    """Post-installation for install mode (pip install .)."""
    def run(self):
        _run_then_symlink(lambda: install.run(self))


setup(
    name="qwen3-vl-vtr",
    version="1.0.0",
    description="PriorTR: Single-Forward Visual Token Reduction for Qwen3-VL",
    author="PriorTR Team",
    packages=find_packages(include=["visual_token_pruning", "visual_token_pruning.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "torchvision",
        "accelerate",
        "qwen-vl-utils==0.0.14",
        "decord",
        "spacy",
    ],
    cmdclass={
        "develop": PostDevelopCommand,
        "install": PostInstallCommand,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
