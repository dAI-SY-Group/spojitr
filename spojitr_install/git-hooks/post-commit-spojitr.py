#!/usr/bin/env python3

import json
import logging
from pathlib import Path
import os
import sys

# *******************************************************************
# CONFIGURATION
# *******************************************************************

LOGGER = logging.getLogger()

SPOJITR_INSTALL_DIR = os.environ.get("SPOJITRPATH")

THIS_FILE_DIR: Path = Path(__file__).parent.absolute()
# bubble up the <VCS_DIR>/.git/hooks/post-commit to the base
VCS_DIR: Path = THIS_FILE_DIR.parent.parent
DOT_GIT_DIR: Path = VCS_DIR / ".git"
SPOJITR_DIR: Path = VCS_DIR / ".spojitr"
SPOJITR_CONFIG_FILE: Path = SPOJITR_DIR / "config.json"
DATABASE_FILE: Path = SPOJITR_DIR / "db.sqlite3"

# *******************************************************************
# FUNCTIONS
# *******************************************************************


def _read_config_file():
    with open(SPOJITR_CONFIG_FILE, "r") as fp:
        return json.load(fp)


def _should_add_issue_id():
    print("Last commit message doesn't contain an Jira issue-id.")

    question = "Do you want to add an issue id [y/n]? "
    answer = None

    while (answer is None) or (answer not in ["y", "Y", "n", "N"]):
        answer = input(question)

    return answer in ["Y", "y"]


def _ask_for_issue_id(issues: list):
    def get_number(question: str):
        answer = None
        while not answer:
            answer = input(question)
            try:
                value = int(answer)
                return value
            except Exception as e:
                answer = None

    n = len(issues)
    print("Make a choice:")
    print()

    for idx, (issue_id, summary, score) in enumerate(issues):
        print(f"({idx+1}) {issue_id:15}: {summary}")

    print()

    answer = -1
    while answer == -1:
        value = get_number(f"Enter 1-{n} to select an issue id, or 0 to abort: ")
        if (0 <= value) and (value <= n):
            answer = value

    return None if answer == 0 else (answer - 1)


# *******************************************************************
# TESTS
# *******************************************************************


def test_main():
    logging.basicConfig(
        format="%(name)s %(levelname)s %(message)s", level=logging.DEBUG
    )

    p: Path = Path(SPOJITR_INSTALL_DIR) / "spojitr"
    sys.path.append(str(p))

    test_ask_for_issue_id()


def test_ask_for_issue_id():
    issue_ids = [
        ("FALCON-994", "summary 994", 0.9),
        ("FALCON-997", "summary 997", 0.8),
        ("FALCON-1234", "summary 1234", 0.5),
    ]
    answer_idx = _ask_for_issue_id(issue_ids)

    if answer_idx is None:
        LOGGER.debug("Aborted selection")
    else:
        LOGGER.debug("Selected issue %s", issue_ids[answer_idx])


# *******************************************************************
# MAIN
# *******************************************************************


def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    LOGGER.debug("=" * 60)
    LOGGER.debug("= Spojitr post commit hook")
    LOGGER.debug("=" * 60)

    if SPOJITR_INSTALL_DIR is None:
        LOGGER.error(
            "Could not locate SPOJITR installation. Is 'SPOJITRPATH' environment variable configured correct?"
        )
        exit(-1)

    if not (DOT_GIT_DIR.exists() and DOT_GIT_DIR.is_dir()):
        LOGGER.error("Commit hook not located in valid git repository: %s", VCS_DIR)
        exit(-1)

    if not DATABASE_FILE.exists():
        LOGGER.error("Spojitr database not found")
        exit(-1)

    p: Path = Path(SPOJITR_INSTALL_DIR) / "spojitr"
    sys.path.append(str(p))

    import spojitr_workflow as wfl

    third_party: Path = Path(SPOJITR_INSTALL_DIR) / "3rd"
    weka_run_script: Path = third_party / "run_weka.py"
    weka_jar = third_party / "weka.jar"

    callbacks = wfl.Callbacks(_should_add_issue_id, _ask_for_issue_id)
    paths = {
        "db_file": DATABASE_FILE,
        "dot_git_dir": DOT_GIT_DIR,
        "spojitr_dir": SPOJITR_DIR,
        "weka_run_script": weka_run_script,
        "weka_jar": weka_jar,
    }

    wfl.run_workflow(callbacks, project_config=_read_config_file(), fs_paths=paths)


if __name__ == "__main__":
    main()
    # test_main()
