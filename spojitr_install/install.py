#!/usr/bin/env python3
"""
Install spojitr toolset
"""

import logging
from pathlib import Path, PosixPath


# *******************************************************************
# CONFIGURATION
# *******************************************************************


logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger()

THIS_FILE_DIR = Path(__file__).parent
INSTALL_DIR = THIS_FILE_DIR


# *******************************************************************
# FUNCTIONS
# *******************************************************************


def install(bashrc_filepath: Path):
    LOGGER.info("Patching %s", bashrc_filepath)

    if "SPOJITRPATH" in open(bashrc_filepath, "r").read():
        LOGGER.warning("spojitr section found in %s... skip patching", bashrc_filepath)
        return

    with open(bashrc_filepath, "a+") as fp:
        content = f"""
# >> spojitr install
export SPOJITRPATH="{str(INSTALL_DIR.absolute())}"

alias spojitr="python3 ${{SPOJITRPATH}}/scripts/spojitr_main.py"
# <<< spojitr install
"""
        fp.write(content)

    LOGGER.info("Updated file %s", bashrc_filepath)


# *******************************************************************
# MAIN
# *******************************************************************


def main():
    LOGGER.info("=" * 60)
    LOGGER.info("= Spojitr installer")
    LOGGER.info("=" * 60)

    bashrc_filepath = PosixPath("~/.bashrc").expanduser()

    if not bashrc_filepath.exists():
        LOGGER.info("Creating file %s", bashrc_filepath)
        bashrc_filepath.touch(mode=0o644, exist_ok=True)

    install(bashrc_filepath)


if __name__ == "__main__":
    main()
