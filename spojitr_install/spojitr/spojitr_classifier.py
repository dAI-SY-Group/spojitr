"""
Utility functions used by the scripts in spojitr installation
"""

import copy
import json
import collections
import logging
import pprint
import typing

from pathlib import Path

import spojitr_database as database
import spojitr_utils

import spojit.evaluation
import spojit.io
import spojit.sample
import spojit.profile
import spojit.weka


# *******************************************************************
# CONFIGURATION
# *******************************************************************


LOGGER = logging.getLogger(__file__)

# taken from: spojit.artifact_filter.DefaultFilter.ISSUE_TYPE_MAPPING
ALL_SPOJIT_ISSUE_TYPES = ["bug", "feature", "improvement", "task"]

TRAIN_ARFF_FILE_NAME = "train_samples.arff"
PREDICT_ARFF_FILE_NAME = "predict_samples.arff"
PREDICT_RESULT_CSV_FILE_NAME = "prediction.csv"
MODEL_FILE_NAME = "model.pmml"


# *******************************************************************
# FUNCTIONS
# *******************************************************************


def _pformat_datasource(ds: spojit.profile.DataSource) -> str:
    """pretty format a datasource

    basically is shortens long lists etc
    """

    def pformat_change_set(cs) -> dict:
        if not cs:
            return None

        new_cs = {k: v for k, v in cs.items() if k != "file_path"}
        file_paths = cs.get("file_path", None)
        if file_paths is not None:
            first_path = file_paths[0]
            new_cs["file_path"] = "{:5d} file paths, 1st: {}".format(
                len(file_paths), first_path
            )

        return new_cs

    n_i = len(ds.issues)
    first_i = ds.issues[0] if n_i > 0 else None

    n_cs = len(ds.change_sets)
    first_cs = ds.change_sets[0] if n_cs > 0 else None
    first_cs_pretty = pformat_change_set(first_cs)

    n_i2cs = len(ds.issue_to_change_set)
    first_i2cs = next(iter(ds.issue_to_change_set.items())) if n_i2cs > 0 else None

    return """DataSource with
  {:5d} issues, 1st: {}
  {:5d} change sets, 1st: {}
  {:5d} links, 1st: {}
""".format(
        n_i, first_i, n_cs, first_cs_pretty, n_i2cs, first_i2cs
    )


def _get_issue_summaries(issue_ids: list, db: database.Database) -> dict:
    def quote(s: str) -> str:
        return f'"{s}"'

    sql_stmt = """
        SELECT issue_id, issue_summary
        FROM issues
        WHERE issue_id in ({ids})
    """.format(
        ids=", ".join(map(quote, issue_ids))
    )

    return {r.issue_id: r.issue_summary for r in db.execute_sql(sql_stmt)}


def _create_spojit_data_source(
    db: database.Database, max_date_time=None, commit_hash_blacklist=None
) -> spojit.profile.DataSource:
    def remap_issue_column(name: str) -> str:
        return "id" if name == "issue_id" else name

    def commit_xform(row_proxy) -> dict:
        """transform row proxy to dict and separate the grouped file paths
        """
        commit = {k: row_proxy[k] for k in commit_column_names}
        commit["file_path"] = row_proxy["grouped_file_paths"].split("<SEP>")
        return commit

    def get_all_issues() -> list:
        sql_stmt = """
            SELECT  {cols}
            FROM    issues
        """.format(
            cols=", ".join(issue_column_names)
        )

        rows = db.execute_sql(sql_stmt)
        issues = [
            dict(
                (remap_issue_column(col_name), row[col_name])
                for col_name in issue_column_names
            )
            for row in rows
        ]
        return issues

    def get_open_issues_or_resolved_before(before_date: str) -> list:
        sql_stmt = """
            SELECT  {cols}
            FROM    issues
            WHERE   (issues.resolved_date IS NULL)
                    OR
                    (issues.resolved_date <= "{before_t}")
        """.format(
            cols=", ".join(issue_column_names), before_t=before_date
        )
        rows = db.execute_sql(sql_stmt)
        issues = [
            dict(
                (remap_issue_column(col_name), row[col_name])
                for col_name in issue_column_names
            )
            for row in rows
        ]
        return issues

    def get_intersection_issues(reference_date: str) -> typing.List[dict]:
        """Calculating the list of issues the exist at 'reference_date' will
        be resolved later.

        To resemble the state at 'reference_date', that status of these issues is set to
        "Open" and the resolution is reset
        """
        sql_stmt = """
            SELECT  {cols}
            FROM    issues
            WHERE   (issues.created_date < "{ref_t}")
                    AND
                    ("{ref_t}" < issues.resolved_date)
        """.format(
            cols=", ".join(issue_column_names), ref_t=reference_date
        )

        issues = []
        for row in db.execute_sql(sql_stmt):
            issue = dict(
                (remap_issue_column(col_name), row[col_name])
                for col_name in issue_column_names
            )

            # the issue is resolved _after_ reference date,
            # so at that point in time it is open
            issue["resolution"] = None
            issue["resolved_date"] = None
            issue["status"] = "Open"
            issues.append(issue)

        return issues

    def get_commits_before(reference_date: str = None, black_list: list = None) -> list:
        commit_hash_blacklist = set(black_list) if black_list else set()
        where_clause = ""

        if reference_date:
            where_clause = f"""WHERE committed_date < "{max_date_time}" """

        sql_stmt = """
            SELECT {columns}, group_concat(file_path, "<SEP>") AS grouped_file_paths
            FROM change_set
            {where}
            GROUP BY commit_hash
        """.format(
            columns=", ".join(commit_column_names), where=where_clause
        )

        change_sets = [
            commit_xform(row)
            for row in db.execute_sql(sql_stmt)
            if row.commit_hash not in commit_hash_blacklist
        ]

        return change_sets

    if max_date_time:
        # LOGGER.warning("Data source filtering till %s", max_date_time)
        pass

    # --- issues ---
    issue_column_names = [
        "issue_id",
        "assignee",
        "assignee_username",
        "status",
        "priority",
        "created_date",
        "type",
        "resolved_date",
        "resolution",
    ]

    if max_date_time:
        issues = get_open_issues_or_resolved_before(max_date_time)
        issues.extend(get_intersection_issues(max_date_time))
    else:
        issues = get_all_issues()

    # --- commits ---
    commit_column_names = [
        "author",
        "author_email",
        "commit_hash",
        "committed_date",
        "commit_msg",
    ]

    change_sets = get_commits_before(
        reference_date=max_date_time, black_list=commit_hash_blacklist
    )

    # --- links ---
    # from all known links, only selected those between artifacts that haven't been filtered
    existing_issue_ids = set(issue["id"] for issue in issues)
    existing_commit_hashes = set(commit["commit_hash"] for commit in change_sets)
    blacklist = set(commit_hash_blacklist) if commit_hash_blacklist else set()

    issue_to_change_sets: dict = collections.defaultdict(list)

    sql_stmt = """
        SELECT DISTINCT issue_id, commit_hash
        FROM   issue_to_change_set
    """

    for row in db.execute_sql(sql_stmt):
        if (row.issue_id not in existing_issue_ids) or (
            row.commit_hash not in existing_commit_hashes
        ):
            # a link between filtered artifacts
            continue

        if row.commit_hash not in blacklist:
            issue_to_change_sets[row.issue_id].append(row.commit_hash)

    return spojit.profile.DataSource(issues, issue_to_change_sets, change_sets)


class IssueToCommitHandler(spojit.io.IssueToOtherSimilarityHandler):
    def __init__(self, db: database.Database):
        sql_stmt = """
            SELECT  issue_id, commit_hash, issue_to_commit_sim AS sim
            FROM    issue_to_commit_similarity
            ORDER BY issue_id, commit_hash
        """
        self._i_2_c: dict = collections.defaultdict(dict)
        for row in db.execute_sql(sql_stmt):
            self._i_2_c[row.issue_id][row.commit_hash] = row.sim

    def get_sim(self, issue_id: str, commit_hash: str):
        issue = self._i_2_c.get(issue_id, None)
        return issue.get(commit_hash, None) if issue else None

    def extend(self, data: typing.Iterable[typing.Tuple]):
        """
        """
        for issue_id, commit_hash, sim in data:
            self._i_2_c[issue_id][commit_hash] = sim


class IssueToCodeHandler(spojit.io.IssueToCodeSimilarityHandler):
    def __init__(self, db: database.Database):
        sql_stmt = """
            SELECT  issue_id, commit_hash, file_path, issue_to_code_sim AS sim
            FROM    issue_to_code_similarity
            ORDER BY issue_id, commit_hash
        """
        self._i_2_c: dict = collections.defaultdict(dict)
        for row in db.execute_sql(sql_stmt):
            self._add(row.issue_id, row.commit_hash, row.file_path, row.sim)

    def get_sim(self, issue_id: str, commit_hash: str, file_path: str):
        issue = self._i_2_c.get(issue_id, None)
        commit = issue.get(commit_hash, None) if issue else None
        return commit.get(file_path, None) if commit else None

    def extend(self, data: typing.Iterable[typing.Tuple]):
        for issue_id, commit_hash, file_path, sim in data:
            self._add(issue_id, commit_hash, file_path, sim)

    def _add(self, issue_id, commit_hash, file_path, sim):
        commit_entry: dict = self._i_2_c[issue_id].get(commit_hash, dict())
        commit_entry[file_path] = sim
        self._i_2_c[issue_id][commit_hash] = commit_entry


def _create_train_profile(
    data_source: spojit.profile.DataSource, profile_file: Path, db: database.Database
):
    """Create a profile for training, i.e. without predictions
    """
    LOGGER.debug("data source ... %s", _pformat_datasource(data_source))

    issue_to_commit_handler = IssueToCommitHandler(db)
    issue_to_code_handler = IssueToCodeHandler(db)

    text_sim = spojit.io.TextSimilarityHandler(
        issue_to_code_lsi=None,
        issue_to_code_vsm_ngram=issue_to_code_handler,
        issue_to_commit_vsm_ngram=issue_to_commit_handler,
    )

    trace_sim = spojit.io.TraceSimilarityHandler(issue_to_file=None)
    to_predict_changsets: list = []

    pg = spojit.profile.ProfileGenerator(
        data_source,
        to_predict_changsets,
        text_similarity_handler=text_sim,
        trace_handler=trace_sim,
        issue_types=ALL_SPOJIT_ISSUE_TYPES[:],
    )

    train_sample_collector = spojit.sample.SampleCollector()
    predict_sample_collector = spojit.sample.SampleCollector()

    LOGGER.debug("Start sampling train data")
    pg.run(train_sample_collector, predict_sample_collector)

    output_data = spojit.sample.create_weka_dataset(
        "train_profile", train_sample_collector.samples
    )

    LOGGER.debug("Start writing arff file")
    generator = spojit.weka.dataset_to_arff(output_data)
    with open(profile_file, "w") as fp:
        fp.writelines("\n".join(generator))
        LOGGER.debug("Wrote file '%s'", profile_file)


def _create_predict_profile(
    commit: dict,
    data_source: spojit.profile.DataSource,
    extra_issue_to_commit_similarities: list,
    extra_issue_to_code_similarities: list,
    profile_file: Path,
    db: database.Database,
):
    # (1) transform commit
    commit = copy.deepcopy(commit)
    commit["file_path"] = [fn for fn, _ in commit["file_infos"]]
    del commit["file_infos"]
    LOGGER.debug("Create predict profile for %s", pprint.pformat(commit))
    LOGGER.debug("predict data source ... %s", _pformat_datasource(data_source))

    issue_to_commit_handler = IssueToCommitHandler(db)
    issue_to_commit_handler.extend(extra_issue_to_commit_similarities)

    issue_to_code_handler = IssueToCodeHandler(db)
    issue_to_code_handler.extend(extra_issue_to_code_similarities)

    text_sim = spojit.io.TextSimilarityHandler(
        issue_to_code_lsi=None,
        issue_to_code_vsm_ngram=issue_to_code_handler,
        issue_to_commit_vsm_ngram=issue_to_commit_handler,
    )

    trace_sim = spojit.io.TraceSimilarityHandler(issue_to_file=None)
    to_predict_changsets: list = [commit]

    pg = spojit.profile.ProfileGenerator(
        data_source,
        to_predict_changsets,
        text_similarity_handler=text_sim,
        trace_handler=trace_sim,
        issue_types=ALL_SPOJIT_ISSUE_TYPES[:],
    )

    train_sample_collector = spojit.sample.SampleCollector()
    predict_sample_collector = spojit.sample.SampleCollector()
    pg.run(train_sample_collector, predict_sample_collector)

    output_data = spojit.sample.create_weka_dataset(
        "predict_profile", predict_sample_collector.samples
    )

    generator = spojit.weka.dataset_to_arff(output_data)
    with open(profile_file, "w") as fp:
        fp.writelines("\n".join(generator))
        LOGGER.debug("Wrote file '%s'", profile_file)


def _train_model_with_weka(
    train_file: Path,
    model_file: Path,
    weka_run_script: Path,
    weka_jar: Path,
    dry_run=False,
):
    args = [
        "python3",
        str(weka_run_script),
        "-e",
        "train",
        "-c",
        "zeroR" if dry_run else "randomForest",
        "--attribute-config=auto",
        "--train-file",
        str(train_file),
        "--test-file",
        str(train_file),
        "--model-file",
        str(model_file),
        "--weka-jar",
        str(weka_jar),
    ]

    if not dry_run:
        args += ["--run"]

    LOGGER.info("Training model ...")
    text = spojitr_utils.capture_command_output(args)
    LOGGER.debug("training result %s", text)


def _predict_with_weka(
    predict_file: Path,
    model_file: Path,
    prediction_output_file: Path,
    weka_run_script: Path,
    weka_jar: Path,
    dry_run=False,
):
    args = [
        "python3",
        str(weka_run_script),
        "-e",
        "predict",
        "--test-file",
        str(predict_file),
        "--model-file",
        str(model_file),
        "--prediction-output-file",
        str(prediction_output_file),
        "--weka-jar",
        str(weka_jar),
    ]

    if not dry_run:
        args += ["--run"]

    LOGGER.debug("Running prediction ...")
    text = spojitr_utils.capture_command_output(args)
    LOGGER.debug("prediction result %s", text)


def _get_top_predictions(prediction_csv_file: Path, top_n=3) -> typing.List[dict]:
    """Parse the predictions file (CSV format) generated by weka and return the top 'top_n' results
    :Returns: list of mappings
                    [{
                        "issue_id": "...",
                        "probability_linked": 0.8
                    }, ...]
    """
    LOGGER.debug("Read prediction_csv_file: %s", str(prediction_csv_file))

    commit_to_pred = spojit.evaluation.order_by_probability(
        spojit.evaluation.group_predictions_by_commit_hash(
            spojit.evaluation.read_prediction_file(prediction_csv_file)
        )
    )

    # we only made predictions for _one_ commit
    assert len(commit_to_pred) == 1
    for commit_hash, entries in commit_to_pred.items():
        return entries[:top_n]

    return []


def _build_issue_choices(
    prediction_csv_file: Path, db: database.Database
) -> typing.List[typing.Tuple]:
    predictions = _get_top_predictions(prediction_csv_file, top_n=3)
    issue_summaries = _get_issue_summaries([e["issue_id"] for e in predictions], db)

    # assemble the result
    result = []
    for prediction in predictions:
        issue_id = prediction["issue_id"]
        result.append(
            (issue_id, issue_summaries[issue_id], prediction["probability_linked"])
        )

    return result


def train(db_file: Path, spojitr_dir: Path, weka_run_script: Path, weka_jar: Path):
    LOGGER.info("Training ... ")

    train_file: Path = spojitr_dir / TRAIN_ARFF_FILE_NAME
    model_file: Path = spojitr_dir / MODEL_FILE_NAME

    db = database.Database(db_file)
    data_source = _create_spojit_data_source(db)
    _create_train_profile(data_source, train_file, db)

    _train_model_with_weka(
        train_file, model_file, weka_run_script, weka_jar, dry_run=False
    )


def predict(
    commit: dict,
    new_issue_to_commit_similarities: list,
    new_issue_to_code_similarities: list,
    db: database.Database,
    spojitr_dir: Path,
    weka_run_script: Path,
    weka_jar: Path,
    file_prefix="",
):
    """Make a prediction for 'commit'

    :param file_prefix: prefix to switch between different files, e.g. for demo mode
    """

    # TODO: is is a good idea to have 'db' as object parameter, or better use a path?
    LOGGER.info("Predicting ... ")

    predict_file: Path = spojitr_dir / (file_prefix + PREDICT_ARFF_FILE_NAME)
    model_file: Path = spojitr_dir / (file_prefix + MODEL_FILE_NAME)
    prediction_result_file = spojitr_dir / (file_prefix + PREDICT_RESULT_CSV_FILE_NAME)

    data_source = _create_spojit_data_source(
        db,
        max_date_time=commit["committed_date"],
        commit_hash_blacklist=[commit["commit_hash"]],
    )

    _create_predict_profile(
        commit,
        data_source,
        new_issue_to_commit_similarities,
        new_issue_to_code_similarities,
        predict_file,
        db,
    )

    _predict_with_weka(
        predict_file,
        model_file,
        prediction_result_file,
        weka_run_script,
        weka_jar,
        dry_run=False,
    )

    issue_choices = _build_issue_choices(prediction_result_file, db)
    LOGGER.debug("Formatted issue choices: %s", issue_choices)
    return issue_choices


# *******************************************************************
# TESTS
# *******************************************************************


THIRD_PARTY = Path("/data/spojitr_install/3rd")
WEKA_RUN_SCRIPT: Path = THIRD_PARTY / "run_weka.py"
WEKA_JAR: Path = THIRD_PARTY / "weka.jar"

FALCON_BASE: Path = Path("~/falcon").expanduser()

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


def test_create_profile():
    config = FALCON_SETUP

    db = database.Database(config["db_file"])
    _create_train_profile(config["spojitr_dir"] / "train.arff", db)


def test_train_model_with_weka():
    config = FALCON_SETUP
    _train_model_with_weka(
        config["spojitr_dir"] / "train.arff",
        config["spojitr_dir"] / "model.pmml",
        WEKA_RUN_SCRIPT,
        WEKA_JAR,
        dry_run=True,
    )


def test_predict_with_weka():
    config = FALCON_SETUP
    _predict_with_weka(
        config["spojitr_dir"] / "predict.arff",
        config["spojitr_dir"] / "model.pmml",
        config["spojitr_dir"] / "predict.csv",
        WEKA_RUN_SCRIPT,
        WEKA_JAR,
        dry_run=True,
    )


def test_build_issue_choices():
    config = CRUNCH_SETUP

    db = database.Database(config["db_file"])
    choices = _build_issue_choices(config["spojitr_dir"] / "demo_prediction.csv", db)
    LOGGER.debug("Choices: %s", choices)


def test_create_spojit_data_source():
    config = FALCON_SETUP
    db = database.Database(config["db_file"])

    complete_ds = _create_spojit_data_source(db)
    LOGGER.debug("Full data source\n%s", _pformat_datasource(complete_ds))

    t = "2017-03-01T00:00:00Z"
    partial_ds = _create_spojit_data_source(db, max_date_time=t)
    LOGGER.debug("Data source at t=%s\n%s", t, _pformat_datasource(partial_ds))


# ---------------------------------------------------------------------------

CRUNCH_BASE: Path = Path("~/crunch").expanduser()

CRUNCH_SETUP = {
    "base": CRUNCH_BASE,
    "dot_git_dir": CRUNCH_BASE / ".git",
    "spojitr_dir": CRUNCH_BASE / ".spojitr",
    "db_file": CRUNCH_BASE / ".spojitr" / "db.sqlite3",
    "project_config": {
        "jiraProjectKey": "CRUNCH",
        "jiraRestUri": "https://issues.apache.org/jira/rest/api/2",
    },
}


def setup_crunch_demo():
    LOGGER.info("Setup Crunch demo")
    # config
    config = CRUNCH_SETUP
    reference_t = "2017-03-01T00:00:00Z"
    train_file: Path = config["spojitr_dir"] / ("demo_" + TRAIN_ARFF_FILE_NAME)
    model_file: Path = config["spojitr_dir"] / ("demo_" + MODEL_FILE_NAME)

    db = database.Database(config["db_file"])
    data_source = _create_spojit_data_source(db, max_date_time=reference_t)

    _create_train_profile(data_source, train_file, db)
    _train_model_with_weka(
        train_file, model_file, WEKA_RUN_SCRIPT, WEKA_JAR, dry_run=False
    )


# *******************************************************************
# MAIN
# *******************************************************************


if __name__ == "__main__":
    logging.basicConfig(
        format="%(name)s %(levelname)s %(message)s", level=logging.DEBUG
    )
    print(f"Hello from {__file__}")

    # test_create_profile()
    # test_train_model_with_weka()
    # test_predict_with_weka()
    test_build_issue_choices()
    # test_create_spojit_data_source()
    # setup_crunch_demo()
