#!/usr/bin/env python3
"""
spojitr
"""

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import sys


# *******************************************************************
# CONFIGURATION
# *******************************************************************


logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger()

SPOJITR_INSTALL_DIR = os.environ.get("SPOJITRPATH")
# CURRENT_DIR needs to be the base directory of a git repository
CURRENT_DIR = Path().absolute()
DOT_GIT_DIR: Path = CURRENT_DIR / ".git"
POST_COMMIT_HOOK_FILE: Path = DOT_GIT_DIR / "hooks" / "post-commit"
POST_COMMIT_HOOK_SPOJITR_FILE: Path = DOT_GIT_DIR / "hooks" / "post-commit-spojitr.py"
SPOJITR_DIR: Path = CURRENT_DIR / ".spojitr"
SPOJITR_CONFIG_FILE: Path = SPOJITR_DIR / "config.json"
DATABASE_FILE: Path = SPOJITR_DIR / "db.sqlite3"


# *******************************************************************
# FUNCTIONS
# *******************************************************************


def _read_config_file():
    with open(SPOJITR_CONFIG_FILE, "r") as fp:
        return json.load(fp)


def _ask_user_data() -> dict:
    project_key, rest_uri = None, None

    while not project_key:
        project_key = input("Jira project key, e.g. HADOOP : ")

    while not rest_uri:
        rest_uri = input(
            "Jira REST URI of hosting server, e.g. https://issues.apache.org/jira/rest/api/2 : "
        )

    return {"jiraProjectKey": project_key, "jiraRestUri": rest_uri}


def _test_server_connectivity(user_data: dict):
    # Test user data
    import spojitr_utils

    LOGGER.info("Test server connectivity ...")

    try:
        jql_query = f"project={user_data['jiraProjectKey']}"
        num_issues = spojitr_utils.jira_get_number_of_search_results(
            jql_query, user_data["jiraRestUri"]
        )
        LOGGER.info(
            "Success, project %s has %d issues.",
            user_data["jiraProjectKey"],
            num_issues,
        )
    except Exception as e:
        LOGGER.error("Could not query number of issues. Error %s", e)
        exit(-1)


def _install_post_commit_hook():
    LOGGER.info("Installing post-commit hook ...")

    hook_source_file: Path = Path(SPOJITR_INSTALL_DIR) / "git-hooks" / "post-commit"
    hook_source_py_file: Path = Path(
        SPOJITR_INSTALL_DIR
    ) / "git-hooks" / "post-commit-spojitr.py"

    msg = f"""\
Existing git post hook file found            : {str(POST_COMMIT_HOOK_FILE.absolute())}
Spojitr requires to install its own hook file: {hook_source_file}

Git only supports one hook file. If multiple ones are required,
the user has to provide the required mechanisms (e.g. meta-script
calling other scripts etc.). see: https://stackoverflow.com/a/36708095
    """

    if POST_COMMIT_HOOK_FILE.exists() and POST_COMMIT_HOOK_FILE.is_file():
        LOGGER.error(msg)
        exit(-1)

    for src, dest in [
        (hook_source_file, POST_COMMIT_HOOK_FILE),
        (hook_source_py_file, POST_COMMIT_HOOK_SPOJITR_FILE),
    ]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        dest.chmod(0o755)
        LOGGER.info("Copied %s to %s", src.absolute(), dest.absolute())


def _check_spojitr_lib_import():
    """Check spojitr implementation scripts can be imported

    We are manipulating the python path to search for spojitr installation
    using SPOJITRPATH, which is configured during installation.

    Usually this works, but may cause troubles
    * On windows: native paths like 'C:\path\subpath\install' vs.
      bash pseudo paths like '/c/path/subpath/install')
    * If path contains spaces, e.g. 'C:\\Users\\John Doe\\...'

    Basically the python pathlib.Path() abstraction, which is consistently
    used within spojitr, should handle all the cases. But we check anyway.
    """

    import os

    # add spojitr to module search path
    p: Path = Path(SPOJITR_INSTALL_DIR) / "spojitr"
    sys.path.append(str(p))

    try:
        import spojitr_utils
    except Exception as e:
        LOGGER.error(
            """\
Spojitr library cannot be imported: %s

Make sure SPOJITRPATH is configured correctly.
Current module search paths are: %s""",
            e,
            sys.path,
        )

        if os.name == "nt":
            # windows
            LOGGER.error(
                """\
-> Make sure SPOJITRPATH is a native, absolute windows path, i.e. starting like this
C:\\."""
            )

        if " " in SPOJITR_INSTALL_DIR:
            LOGGER.error("-> SPOJITRPATH needs to be quoted when path contains spaces.")

        import traceback

        LOGGER.error("\n\n%s", traceback.format_exc())
        exit(-1)


# *******************************************************************
# COMMANDS
# *******************************************************************


def _commands_help_message():
    return """
sub commands:
  build-db      Build local artifact database used as cache
                for spojitr operations

  init          Create an empty spojitr project

  train-model   Train spojit classifier on local artifact database
"""


def init_cmd(args):
    if SPOJITR_CONFIG_FILE.exists() and SPOJITR_CONFIG_FILE.is_file():
        LOGGER.info("Project already configured (config file %s)", SPOJITR_CONFIG_FILE)

        config = _read_config_file()

        LOGGER.info("Current configuration\n")
        print(json.dumps(config, indent=2))
        return

    user_data = _ask_user_data()
    _test_server_connectivity(user_data)
    _install_post_commit_hook()

    SPOJITR_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPOJITR_CONFIG_FILE, "w") as fp:
        json.dump(user_data, fp)

    LOGGER.info("Initialized empty spojitr repository in %s", SPOJITR_DIR.absolute())


def build_db_cmd(args):
    if not (SPOJITR_CONFIG_FILE.exists() and SPOJITR_CONFIG_FILE.is_file()):
        LOGGER.error("Git repository is not configured with spojitr")
        exit(-1)

    import spojitr_database_ops as db_ops

    project_config = _read_config_file()

    db_ops.populate_database(
        project_config,
        db_file=DATABASE_FILE,
        spojitr_dir=SPOJITR_DIR,
        dot_git_dir=DOT_GIT_DIR,
    )


def train_model_cmd(args):
    if not (DATABASE_FILE.exists() and DATABASE_FILE.is_file()):
        LOGGER.error("Spojitr database not found. Expected location %s)", DATABASE_FILE)
        exit(-1)

    import spojitr_classifier

    third_party: Path = Path(SPOJITR_INSTALL_DIR) / "3rd"
    weka_run_script: Path = third_party / "run_weka.py"
    weka_jar = third_party / "weka.jar"

    spojitr_classifier.train(
        db_file=DATABASE_FILE,
        spojitr_dir=SPOJITR_DIR,
        weka_run_script=weka_run_script,
        weka_jar=weka_jar,
    )


# *******************************************************************
# MAIN
# *******************************************************************


def main():
    parser = argparse.ArgumentParser(
        prog="spojitr",
        description="Spojitr workflow engine for git and jira",
        epilog=_commands_help_message(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        type=str,
        choices=["init", "build-db", "train-model"],
        help="sub command (description below)",
    )
    parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="silent mode, i.e. minimal commandline logging",
    )
    args = parser.parse_args()

    if args.silent:
        logging.getLogger().setLevel(logging.ERROR)

    if SPOJITR_INSTALL_DIR is None:
        LOGGER.error(
            "Could not locate SPOJITR installation. Is 'SPOJITRPATH' environment variable configured correctly?"
        )
        exit(-1)

    if not (DOT_GIT_DIR.exists() and DOT_GIT_DIR.is_dir()):
        LOGGER.error("Current directory does not look like a git repository")
        exit(-1)

    # try to pull in spojitr implementation
    _check_spojitr_lib_import()

    if args.command == "init":
        init_cmd(args)
    elif args.command == "build-db":
        build_db_cmd(args)
    elif args.command == "train-model":
        train_model_cmd(args)


if __name__ == "__main__":
    main()
