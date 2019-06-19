"""
spojitr workflow
"""

import copy
import json
import logging
import re
import pprint
import typing

from pathlib import Path

import spojitr_classifier
import spojitr_database as database
import spojitr_database_ops as database_ops
import spojitr_text_processing as text_processing
import spojitr_utils

# *******************************************************************
# CONFIGURATION
# *******************************************************************

LOGGER = logging.getLogger()

DEMO_FILE_NAME = "demo.json"

# *******************************************************************
# FUNCTIONS
# *******************************************************************


def _get_demo_config(spojitr_dir: Path) -> typing.Optional[dict]:
    try:
        with open(spojitr_dir / DEMO_FILE_NAME, "r") as fp:
            config = json.load(fp)
            lvl = config.get("nesting_level", 0)
            config["nesting_level"] = lvl
            return config
    except:
        return None


def _write_demo_config(config: dict, spojitr_dir: Path):
    cfg = copy.deepcopy(config)

    if ("nesting_level" in cfg) and (cfg["nesting_level"] == 0):
        del cfg["nesting_level"]
    try:
        with open(spojitr_dir / DEMO_FILE_NAME, "w") as fp:
            json.dump(cfg, fp, indent=2)
    except:
        pass


def _calculate_issue_candidates(
    committed_date: str, db: database.Database
) -> typing.List[str]:
    """Calculate list of issue ids (that have a description) for a commit

    :param committed_date: iso date of the commit
    :param db: database
    :Returns: list of issue ids. All corresponding issues _have_ a description
    """
    ics = database_ops.IssueCandidateLookup(db)
    issue_candidates = list(ics.get_candidates_for_unlinked_commit(committed_date))
    LOGGER.debug("Issue candidates %d", len(issue_candidates))

    sql_stmt = """
        SELECT issue_id
        FROM issues
        WHERE issue_description IS NOT NULL
    """
    issue_ids_that_have_a_description = set(
        (r.issue_id for r in db.execute_sql(sql_stmt))
    )

    issue_candidates_with_description = [
        issue_id
        for issue_id in issue_candidates
        if issue_id in issue_ids_that_have_a_description
    ]

    return issue_candidates_with_description


def _issue_to_commit_similarities(
    commit_hash: str, commit_msg: str, issue_data: typing.List[typing.Tuple]
) -> typing.List[typing.Tuple]:
    """
    :param issue_data: list of pairs (<issue id>, <issue message>)
    :Returns: list of similarities
    """
    # TODO: consider moving this code to database_ops
    issue_to_commit_similarity: typing.List[dict] = []

    if commit_msg is not None:
        query_doc = commit_msg

        issue_ids = [issue_id for issue_id, _ in issue_data]
        corpus_docs = [msg for _, msg in issue_data]

        try:
            similarities = text_processing.calculate_similarity(query_doc, corpus_docs)
            for issue_id, similarity in zip(issue_ids, similarities):
                issue_to_commit_similarity.append((issue_id, commit_hash, similarity))
        except ValueError as ve:
            # possibly empty vocabulary error
            pass

    return issue_to_commit_similarity


def _issue_to_code_similarities(
    commit_hash: str,
    issue_data: typing.List[typing.Tuple],
    code_data: typing.List[typing.Tuple],
) -> typing.List[typing.Tuple]:
    """
    :param issue_data: list of pairs (<issue id>, <issue message>)
    :param code data: list of pairs (<file path>, <file content>)
    :Returns: list of similarities
    """
    # TODO: consider moving this code to database_ops
    issue_to_code_similarity = []

    issue_ids = [issue_id for issue_id, _ in issue_data]
    query_docs = [msg for _, msg in issue_data]

    file_paths = [fn for fn, _ in code_data]
    file_contents = [t for _, t in code_data]

    corpus_docs = zip(file_paths, file_contents)

    try:
        simi_calculator = text_processing.CosineSimilarity(corpus_docs)
    except ValueError as ve:
        # possibly empty vocabulary error
        return issue_to_code_similarity

    #  rows are the docs (files), columns are the queries (issue descriptions)
    similarity_matrix = simi_calculator.get_similarities(query_docs)
    assert similarity_matrix.shape == (len(file_paths), len(issue_ids))

    for col_idx, issue_id in enumerate(issue_ids):
        similarities = similarity_matrix[:, col_idx].flatten()
        # store in result structure
        for file_path, similarity in zip(file_paths, similarities):
            issue_to_code_similarity.append(
                (issue_id, commit_hash, file_path, similarity)
            )

    return issue_to_code_similarity


def _db_get_commit_info(commit_hash: str, db: database.Database) -> dict:
    sql_stmt = """
    SELECT  commit_hash,
            author,
            author_email AS email,
            committed_date AS `date`,
            commit_msg AS msg,
            group_concat(file_path, "<SEP>") AS grouped_file_paths
    FROM change_set
    WHERE commit_hash = "{hash}"
    """.format(
        hash=commit_hash
    )

    row_proxy = db.execute_sql(sql_stmt).fetchone()
    commit = {k: row_proxy[k] for k, _ in row_proxy.items()}
    commit["paths"] = row_proxy["grouped_file_paths"].split("<SEP>")
    del commit["grouped_file_paths"]
    return commit


def _git_get_latest_commit(dot_git_dir) -> dict:
    commit_hash = spojitr_utils.git_get_latest_commit_hash(dot_git_dir)
    commit_info = spojitr_utils.git_get_commit_information(commit_hash, dot_git_dir)
    commit_info["commit_hash"] = commit_hash
    return commit_info


class DatabaseUpdater:
    def __init__(self):
        self.preprocessed_commit = None
        self.issue_to_change_set_links = []
        self.issue_to_commit_similarities = []
        self.issue_to_code_similarities = []

    def execute(self, db: database.Database):
        LOGGER.debug("Update database")
        self._execute_impl(db)

    def _execute_impl(self, db: database.Database):
        if self.preprocessed_commit:
            LOGGER.debug(
                "Add new commit:\n%s\n", pprint.pformat(self.preprocessed_commit)
            )
            db.insert_change_set_entry(**self.preprocessed_commit)

        if self.issue_to_change_set_links:
            LOGGER.debug(
                "Add new I->C links:\n%s\n",
                pprint.pformat(self.issue_to_change_set_links),
            )
            db.insert_issue_to_changeset_links(self.issue_to_change_set_links)

        if self.issue_to_commit_similarities:
            LOGGER.debug(
                "Add new I2Commit similarities:\n%s\n",
                pprint.pformat(self.issue_to_commit_similarities),
            )
            db.insert_issue_to_commit_similarities(self.issue_to_commit_similarities)

        if self.issue_to_code_similarities:
            LOGGER.debug(
                "Add new I2Code similarities:\n%s\n",
                pprint.pformat(self.issue_to_code_similarities),
            )
            db.insert_issue_to_code_similarities(self.issue_to_code_similarities)


def _case_update_issues(db: database.Database, project_config: dict) -> list:
    LOGGER.debug("Case: Update issues")

    # TODO: what about updating commits? Maybe there is new commit data as well because
    # of `git pull` etc

    return database_ops._update_jira_issues(db, project_config)


def _case_commit_with_existing_issue_id(
    issue_ids_in_commit_msg: set, commit: dict, db: database.Database, dot_git_dir: Path
) -> DatabaseUpdater:
    """

    :param commit_info: commit structure
                    {
                        "commit_hash": "...",
                        "author": "John Doe",
                        "email": "jdoe@cyborg.org",
                        "date": "2018-12-24T13:23:26Z",
                        "msg": "MAVEN-23 message with id",
                        "paths": ["text_file1.txt", "/path/text_file2.txt"],
                    }
    """

    def quote(s):
        return f'"{s}"'

    issue_ids_in_commit_msg = set(issue_ids_in_commit_msg)
    LOGGER.debug("Case: commit with existing issue ids %s", issue_ids_in_commit_msg)

    sql_stmt = """
        SELECT issue_id, issue_description
        FROM issues
        WHERE issue_id in ({issues})
    """.format(
        issues=", ".join(map(quote, issue_ids_in_commit_msg))
    )

    found_issues = [(r.issue_id, r.issue_description) for r in db.execute_sql(sql_stmt)]
    found_issue_ids_in_database = [issue_id for issue_id, _ in found_issues]
    unknown_ids = set(issue_ids_in_commit_msg) - set(found_issue_ids_in_database)

    if unknown_ids:
        # TODO FIXME: what if the there is an issue id, i.e. valid format
        #           but does not point to an existing issue?
        #           What if all found issue Ids aer invalid?
        #           Maybe ask whether this is correct or a mistake?
        LOGGER.warning(
            "Commit message contains unknown issue ids: %s", ", ".join(unknown_ids)
        )

    commit_hash = commit["commit_hash"]
    # (1) update commit
    preprocessed_commit = database_ops.preprocess_commit(
        commit_hash, commit, dot_git_dir
    )
    # LOGGER.debug("Preprocessed %s", preprocessed_commit)

    # (2) update issue to change set links
    issue_to_change_set_links = [
        (issue_id, commit_hash) for issue_id in found_issue_ids_in_database
    ]

    # Prepare similarities
    issue_data = [(issue_id, issue_msg) for issue_id, issue_msg in found_issues]

    # (3) issue to change set similarities
    issue_to_commit_similarity = _issue_to_commit_similarities(
        commit_hash, preprocessed_commit["commit_msg"], issue_data
    )

    # (4) issue to file path similarities
    code_data = [(fn, text) for fn, text in preprocessed_commit["file_infos"]]
    issue_to_code_similarity = _issue_to_code_similarities(
        commit_hash, issue_data, code_data
    )

    dbu = DatabaseUpdater()
    dbu.preprocessed_commit = preprocessed_commit
    dbu.issue_to_change_set_links = issue_to_change_set_links
    dbu.issue_to_commit_similarities = issue_to_commit_similarity
    dbu.issue_to_code_similarities = issue_to_code_similarity
    return dbu


def _case_reject_adding_issue_id(
    commit: dict, db: database.Database, dot_git_dir: Path
) -> DatabaseUpdater:
    """
    :param commit_info: commit structure
                    {
                        "commit_hash": "...",
                        "author": "John Doe",
                        "email": "jdoe@cyborg.org",
                        "date": "2018-12-24T13:23:26Z",
                        "msg": "message",
                        "paths": ["text_file1.txt", "/path/text_file2.txt"],
                    }
    """
    LOGGER.debug("reject adding issue id for %s", pprint.pformat(commit))

    commit_hash = commit["commit_hash"]
    # (1) update commit

    # TODO: same as _case_commit_with_existing_issue_id
    preprocessed_commit = database_ops.preprocess_commit(
        commit_hash, commit, dot_git_dir
    )

    # (2) No links -> no update of table

    # Prepare similarities
    issue_candidates_with_description = _calculate_issue_candidates(
        preprocessed_commit["committed_date"], db
    )
    issue_descriptions = database_ops.get_issue_descriptions(db)

    issue_data = [
        (issue_id, issue_descriptions[issue_id])
        for issue_id in issue_candidates_with_description
    ]

    # (3) issue to change set similarities
    issue_to_commit_similarity = _issue_to_commit_similarities(
        commit_hash, preprocessed_commit["commit_msg"], issue_data
    )

    # (4) issue to code similarities
    code_data = [(fn, text) for fn, text in preprocessed_commit["file_infos"]]
    issue_to_code_similarity = _issue_to_code_similarities(
        commit_hash, issue_data, code_data
    )

    dbu = DatabaseUpdater()
    dbu.preprocessed_commit = preprocessed_commit
    dbu.issue_to_commit_similarities = issue_to_commit_similarity
    dbu.issue_to_code_similarities = issue_to_code_similarity
    return dbu


def _case_reject_predicted_issue_id(
    commit: dict, db: database.Database, dot_git_dir: Path
) -> DatabaseUpdater:
    LOGGER.debug("case_reject_predicted_issue_id")
    return _case_reject_adding_issue_id(commit, db, dot_git_dir)


def _case_amend_last_commit(
    issue_id: str, db: database.Database, fs_paths: dict
) -> DatabaseUpdater:
    LOGGER.debug("case_amend_last_commit with %s", issue_id)

    if len(spojitr_utils.find_all_jira_identifiers(issue_id)) != 1:
        LOGGER.error("Error amening last commit. Invalid issue id '%s'", issue_id)
        return

    latest_commit_info = _git_get_latest_commit(fs_paths["dot_git_dir"])
    latest_commit_message = latest_commit_info["msg"]
    new_message = f"{issue_id} {latest_commit_message}"

    LOGGER.debug("Modified message: %s", new_message)

    spojitr_utils.git_amend_last_commit(new_message, fs_paths["dot_git_dir"])

    # IMPORTANT: git commit --amend triggers the post-commit hook again,
    #   but this time with existing commit info and all the new data is written to the
    #   database
    #
    #   Code beyond this comment runs _AFTER_ the (nested) commit hook has completed

    LOGGER.debug(
        "= code after git commit --amend, i.e. after nested commit hook has completed ="
    )

    return DatabaseUpdater()


def _get_predictions(
    commit: dict,
    db: database.Database,
    fs_paths: dict,
    demo_mode: typing.Optional[dict] = None,
) -> typing.List[typing.Tuple]:
    """Create predictions for a commit

    :Returns: list of choices
                [
                    (<issue-id>, <issue-summary>, <score>),
                    ...
                ]
    """
    LOGGER.debug("get predictions paths %s", fs_paths)
    LOGGER.debug("Generating list with possible issue-ids ...")

    # At this point, that database contains all issue data. Thus we can create
    # a data source for spojit from this database, but we have to calculate
    # the different similarities

    commit_hash = commit["commit_hash"]
    preprocessed_commit = database_ops.preprocess_commit(
        commit_hash, commit, fs_paths["dot_git_dir"]
    )

    # (1) create datasource

    # prepare similarities
    issue_candidates_with_description = _calculate_issue_candidates(
        preprocessed_commit["committed_date"], db
    )
    issue_descriptions = database_ops.get_issue_descriptions(db)

    issue_data = [
        (issue_id, issue_descriptions[issue_id])
        for issue_id in issue_candidates_with_description
    ]

    # (3) issue to change set similarities
    issue_to_commit_similarity = _issue_to_commit_similarities(
        commit_hash, preprocessed_commit["commit_msg"], issue_data
    )

    # (4) issue to code similarities
    code_data = [(fn, text) for fn, text in preprocessed_commit["file_infos"]]

    LOGGER.debug("I2code issue data %s", pprint.pformat(issue_data))
    LOGGER.debug("I2code code data %s", pprint.pformat(code_data))
    LOGGER.debug("prepocessed %s", pprint.pformat(preprocessed_commit))

    issue_to_code_similarity = _issue_to_code_similarities(
        commit_hash, issue_data, code_data
    )
    LOGGER.debug(
        "New issue to code similarities #%s",
        pprint.pformat(len(issue_to_code_similarity)),
    )

    # (5) run classifier
    prefix = ""
    if demo_mode is not None:
        LOGGER.debug("DEMO PREDICTION")
        prefix = "demo_"

    issue_choices = spojitr_classifier.predict(
        preprocessed_commit,
        issue_to_commit_similarity,
        issue_to_code_similarity,
        db,
        fs_paths["spojitr_dir"],
        fs_paths["weka_run_script"],
        fs_paths["weka_jar"],
        file_prefix=prefix,
    )

    return issue_choices


class Callbacks:
    """Workflow callbacks
    """

    def __init__(
        self,
        func_query_add_issue_id: typing.Callable[[], bool],
        func_query_issue_id: typing.Callable[
            [typing.List[typing.Tuple]], typing.Optional[int]
        ],
    ):
        """
        :param func_query_add_issue_id: Callback function to ask, whether the user wants to add an issue id
                                        The function has no args and returns a boolean

        :param func_query_issue_id: Callback to select an issue from a list
                                    The function argument is a list of n choices. A choice is a tuple of
                                    (<issue-id>, <issue-description>, <score>).
                                    The return value is either an integer in range 0 .. (n-1) to select a choice,
                                    or 'None' if selection was aborted
        """
        self._func_query_add_issue_id = func_query_add_issue_id
        self._func_query_issue_id = func_query_issue_id


def _run_workflow_impl(
    latest_commit_info,
    callbacks: Callbacks,
    project_config: dict,
    db: database.Database,
    fs_paths: dict,
    demo_mode: typing.Optional[dict] = None,
) -> DatabaseUpdater:
    LOGGER.debug("Run workflow impl")
    LOGGER.debug("Latest commit info %s", latest_commit_info)

    latest_commit_message = latest_commit_info["msg"]
    issue_ids_in_commit_msg = set(
        spojitr_utils.find_all_jira_identifiers(latest_commit_message)
    )
    if issue_ids_in_commit_msg:
        return _case_commit_with_existing_issue_id(
            issue_ids_in_commit_msg, latest_commit_info, db, fs_paths["dot_git_dir"]
        )

    if not callbacks._func_query_add_issue_id():
        return _case_reject_adding_issue_id(
            latest_commit_info, db, fs_paths["dot_git_dir"]
        )

    predictions = _get_predictions(latest_commit_info, db, fs_paths, demo_mode)
    LOGGER.debug("Predictions %s", pprint.pformat(predictions))

    selected_index = callbacks._func_query_issue_id(predictions)
    if selected_index is None:
        return _case_reject_predicted_issue_id(
            latest_commit_info, db, fs_paths["dot_git_dir"]
        )

    selected_issue_id = predictions[selected_index][0]
    return _case_amend_last_commit(selected_issue_id, db, fs_paths)


# *******************************************************************
# PUBLIC FUNCTIONS
# *******************************************************************


def run_workflow(callbacks: Callbacks, project_config: dict, fs_paths: dict):
    """
    :param fs_paths: file system paths
    """
    LOGGER.debug("Starting spojitr workflow")
    LOGGER.debug("path setup %s", pprint.pformat(fs_paths))

    demo_mode = _get_demo_config(fs_paths["spojitr_dir"])

    if demo_mode:
        demo_mode["nesting_level"] += 1
        _write_demo_config(demo_mode, fs_paths["spojitr_dir"])

        if demo_mode["nesting_level"] == 1:
            LOGGER.warning(
                """\
=================================================
= SPOJITR DEMO MODE                             =
================================================="""
            )

    db = database.Database(fs_paths["db_file"])
    git_latest_commit_info = _git_get_latest_commit(fs_paths["dot_git_dir"])

    if demo_mode:
        # modify commit info to match the original author, date, ...
        db_commit_info = _db_get_commit_info(demo_mode["commit_hash"], db)

        # modify essential fields
        latest_commit_info = copy.deepcopy(git_latest_commit_info)
        latest_commit_info["commit_hash"] = db_commit_info["commit_hash"]
        latest_commit_info["author"] = db_commit_info["author"]
        latest_commit_info["email"] = db_commit_info["email"]
        latest_commit_info["date"] = db_commit_info["date"]
        # TODO: sanity check: db filepaths and git file paths should match!

        LOGGER.debug(
            "DEMO MODE: changed %s -> %s",
            pprint.pformat(git_latest_commit_info),
            pprint.pformat(latest_commit_info),
        )
    else:
        latest_commit_info = git_latest_commit_info

    # -- Begin core workflow
    if not demo_mode:
        _case_update_issues(db, project_config)

    db_updater = _run_workflow_impl(
        latest_commit_info,
        callbacks,
        project_config,
        db=db,
        fs_paths=fs_paths,
        demo_mode=demo_mode,
    )

    if not demo_mode:
        db_updater.execute(db)
    # -- End core workflow

    if demo_mode:
        demo_mode["nesting_level"] -= 1
        _write_demo_config(demo_mode, fs_paths["spojitr_dir"])


# *******************************************************************
# TESTS
# *******************************************************************


FALCON_BASE: Path = Path("/root/falcon")

FALCON_SETUP = {
    "base": FALCON_BASE,
    "dot_git_dir": FALCON_BASE / ".git",
    "spojitr_dir": FALCON_BASE / ".spojitr",
    "db_file": FALCON_BASE / ".spojitr" / "falcon.sqlite3",
    "project_config": {
        "jiraProjectKey": "FALCON",
        "jiraRestUri": "https://issues.apache.org/jira/rest/api/2",
    },
}

FALCON_COMMIT_WITH_ISSUE_ID = {
    "author": "pallavi-rao",
    "commit_hash": "470e5e9f5de9ba1b6149dec60e87d3a04270eda3",
    "date": "2018-08-09T16:02:37Z",
    "email": "pallavi.rao@inmobi.com",
    "msg": "FALCON-2341 Entity SLA Alert publishing wrong results into DB",
    "paths": [
        "common/src/main/java/org/apache/falcon/persistence/EntitySLAAlertBean.java",
        "prism/src/main/java/org/apache/falcon/jdbc/MonitoringJdbcStateStore.java",
        "prism/src/main/java/org/apache/falcon/service/EntitySLAAlertService.java",
        "prism/src/main/java/org/apache/falcon/service/EntitySLAMonitoringService.java",
    ],
}

FALCON_COMMIT_WITHOUT_ISSUE_ID = FALCON_COMMIT_WITH_ISSUE_ID.copy()
FALCON_COMMIT_WITHOUT_ISSUE_ID[
    "msg"
] = "Entity SLA Alert publishing wrong results into DB"


def test_get_latest_commit():
    latest_commit = _git_get_latest_commit(FALCON_SETUP["dot_git_dir"])
    LOGGER.debug("Latest commit %s", pprint.pformat(latest_commit))


def test_run_work_flow_impl():
    def show_predictions_handler(predictions: list) -> int:
        LOGGER.debug("=== start predictions handler ===")
        for idx, p in enumerate(predictions):
            LOGGER.debug("(%d): %s", idx, p)
        LOGGER.debug("=== end predictions handler ===")

        return 1

    db = database.Database(FALCON_SETUP["db_file"])

    callbacks_add_issue = Callbacks(lambda: True, show_predictions_handler)
    callbacks_reject_issue = Callbacks(lambda: False, lambda arg: None)

    fs_paths = {k: FALCON_SETUP[k] for k in ["dot_git_dir", "spojitr_dir", "db_file"]}

    third_party: Path = Path("/data/spojitr_install") / "3rd"
    fs_paths["weka_run_script"] = third_party / "run_weka.py"
    fs_paths["weka_jar"] = third_party / "weka.jar"

    _run_workflow_impl(
        FALCON_COMMIT_WITHOUT_ISSUE_ID,
        callbacks_add_issue,
        FALCON_SETUP["project_config"],
        db,
        fs_paths,
    )


def test_case_commit_with_existing_issue_id():
    db = database.Database(FALCON_SETUP["db_file"])

    latest_commit = _git_get_latest_commit(FALCON_SETUP["dot_git_dir"])
    dbu = _case_commit_with_existing_issue_id(
        ["FALCON-2341", "FOO-BAR", "FALCON-2341"],
        latest_commit,
        db,
        FALCON_SETUP["dot_git_dir"],
    )

    dbu.execute()


def test_case_reject_adding_issue_id():
    db = database.Database(FALCON_SETUP["db_file"])
    latest_commit = _git_get_latest_commit(FALCON_SETUP["dot_git_dir"])

    dbu = _case_reject_adding_issue_id(latest_commit, db, FALCON_SETUP["dot_git_dir"])
    dbu.execute()


def test_get_demo_mode():
    config = _get_demo_config(Path("~/falcon/.spojitr").expanduser())
    if config:
        LOGGER.info("DEMO MODE %s", pprint.pformat(config))
    else:
        LOGGER.info("NORMAL MODEL")


def test():
    lc = _git_get_latest_commit(Path("~/falcon/.git").expanduser())
    LOGGER.info("LC %s", pprint.pformat(lc))
    LOGGER.info("fake %s", pprint.pformat(FALCON_COMMIT_WITH_ISSUE_ID))

    db = database.Database(Path("~/falcon/.spojitr/db.sqlite3").expanduser())
    info = _db_get_commit_info("00a2b3a95aee3fc68a8adf3f04c988df205fe4fe", db)
    LOGGER.info("query %s", pprint.pformat(info))


# *******************************************************************
# MAIN
# *******************************************************************


if __name__ == "__main__":
    logging.basicConfig(
        format="%(name)s %(levelname)s %(message)s", level=logging.DEBUG
    )
    print(f"Hello from {__file__}")
    # test_get_latest_commit()
    # test_case_commit_with_existing_issue_id()
    test_case_reject_adding_issue_id()
    # test_run_work_flow_impl()
    # test_get_demo_mode()
    # test()
