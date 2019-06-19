#!/usr/bin/env python3
"""
spojitr demo for project crunch
"""

import logging
import os
import sys

from pathlib import Path

# *******************************************************************
# CONFIGURATION
# *******************************************************************


logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger()

SPOJITR_INSTALL_DIR = os.environ.get("SPOJITRPATH")

CRUNCH_BASE: Path = Path("~/crunch").expanduser()
CRUNCH_CONFIG = {
    "base": CRUNCH_BASE,
    "dot_git_dir": CRUNCH_BASE / ".git",
    "spojitr_dir": CRUNCH_BASE / ".spojitr",
    "db_file": CRUNCH_BASE / ".spojitr" / "db.sqlite3",
    "train_file": CRUNCH_BASE / ".spojitr" / "demo_train_samples.arff",
    "model_file": CRUNCH_BASE / ".spojitr" / "demo_model.pmml",
}


# *******************************************************************
# FUNCTIONS
# *******************************************************************


def _prepare_crunch(weka_run_script: Path, weka_jar: Path):
    import spojitr_classifier as classifier
    import spojitr_database as database

    reference_t = "2017-03-01T00:00:00Z"

    LOGGER.info("=" * 40)
    LOGGER.info(
        """\
Setup demo for project "crunch"

   reference training date: %s
""",
        reference_t,
    )
    LOGGER.info("=" * 40)

    db = database.Database(CRUNCH_CONFIG["db_file"])
    data_source = classifier._create_spojit_data_source(db, max_date_time=reference_t)

    classifier._create_train_profile(data_source, CRUNCH_CONFIG["train_file"], db)
    classifier._train_model_with_weka(
        CRUNCH_CONFIG["train_file"],
        CRUNCH_CONFIG["model_file"],
        weka_run_script,
        weka_jar,
        dry_run=False,
    )


# *******************************************************************
# MAIN
# *******************************************************************


def main():
    if SPOJITR_INSTALL_DIR is None:
        LOGGER.error(
            "Could not locate SPOJITR installation. Is 'SPOJITRPATH' environment variable configured correctly?"
        )
        exit(-1)

    p: Path = Path(SPOJITR_INSTALL_DIR) / "spojitr"
    sys.path.append(str(p))

    third_party = Path(SPOJITR_INSTALL_DIR) / "3rd"
    weka_run_script: Path = third_party / "run_weka.py"
    weka_jar: Path = third_party / "weka.jar"

    _prepare_crunch(weka_run_script, weka_jar)


if __name__ == "__main__":
    main()
