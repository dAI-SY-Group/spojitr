"""
Spojitr database operations
"""

import collections
from joblib import Parallel, delayed
import datetime
import logging
from pathlib import Path
import subprocess
import tqdm
import typing

import spojit.artifact_filter
import spojit.profile

# spojitr package
import spojitr_database as database
import spojitr_utils
import spojitr_text_processing as text_processing


# *******************************************************************
# CONFIGURATION
# *******************************************************************


LOGGER = logging.getLogger()


# *******************************************************************
# FUNCTIONS
# *******************************************************************

_is_source_file = spojit.artifact_filter.DefaultFilter.is_source_file


def preprocess_commit(commit_hash: str, commit_info: dict, dot_git_dir: Path):
    """Preprocess commit for storage in database

    :param commit_hash: commit hash
    :param commit_info: structure with basic information about commit, including
                        changed file paths
    :param dot_git_dir: required to get file contents
    :Returns: processed commit structure
                {
                    "commit_hash": "",
                    "author": "..."
                    // ....
                    "file_infos": [
                        ("path/file1.java", "text of file 1"),
                        ("path/file2.java", "text of file 2"),
                    ]
                }
    """
    LOGGER.debug("Preprocess commit %s", commit_info)

    data = {
        "commit_hash": commit_hash,
        "author": commit_info["author"],
        "author_email": commit_info["email"],
        "committed_date": commit_info["date"],
        "commit_msg": text_processing.preprocess(commit_info["msg"]),
    }

    file_infos = []
    # HACK: pure evil: exploit spojit internals by filtering the paths. But this saves a lot of otherwise
    # time consuming text preprocessing and database storage
    for file_path in filter(_is_source_file, commit_info["paths"]):
        file_content = spojitr_utils.git_get_file_content(
            commit_hash, file_path, dot_git_dir
        )
        file_content_processed = text_processing.preprocess(file_content)
        file_infos.append((file_path, file_content_processed))

    data["file_infos"] = file_infos
    return data


def _store_commit(preprocessed_commit: dict, db: database.Database):
    """
    .. see also:: preprocess_commit()
    """
    LOGGER.debug("Store commit %s", preprocessed_commit["commit_hash"])
    db.insert_change_set_entry(**preprocessed_commit)


def _fetch_all_commits(db: database.Database, spojitr_dir: Path, dot_git_dir: Path):
    """Query projects VCS and extrace basic information like author, email, modified files etc

    The results are stored in db

    :param db: spojitr database
    """

    def preprocess_task(args):
        commit_hash, dot_git_dir = args

        commit_info = spojitr_utils.git_get_commit_information(commit_hash, dot_git_dir)

        preprocessed_commit = preprocess_commit(commit_hash, commit_info, dot_git_dir)
        return preprocessed_commit

    LOGGER.info("Fetch git commits ...")
    db.delete_table_contents("change_set")

    common_args = ["git", f"--git-dir={dot_git_dir}"]

    ###################################
    # capture all commit hashes in file
    args = common_args + ["log", "--all", "--pretty=%H", "--diff-filter=d"]
    LOGGER.debug("Execute cmd: %s", " ".join(args))

    # create temporary file containing all commit hashes
    commit_hashes_file: Path = spojitr_dir / "commit_hashes.txt"
    with open(commit_hashes_file, "w") as fp:
        subprocess.call(args, stdout=fp)

    args = common_args + ["rev-list", "--count", "HEAD", "--all"]
    total_number_of_commits = int(spojitr_utils.capture_command_output(args))

    # process each commit and extract basic information
    if True:
        ## PARALLEL VERSION
        # FIXME: determine good value for 'n_jobs'. The documented setting '-1'
        #        does not work as expected, at least within docker container
        with Parallel(n_jobs=3) as parallel, tqdm.tqdm(
            total=total_number_of_commits,
            desc="Fetching commits",
            bar_format=spojitr_utils.TQDM_FORMAT,
            disable=LOGGER.getEffectiveLevel() > logging.INFO,
        ) as pbar:
            # algorithm: split all commit hashes in chunks and process the chunks in parallel
            #            The chunking is only required to provide feedback for the user, i.e.
            #            updating the progress bar
            commit_hashes = spojitr_utils.linewise(commit_hashes_file)
            chunk_size = 100

            for commit_hash_chunk in spojitr_utils.grouper(chunk_size, commit_hashes):
                # (1) process the commits in the chunk in parallel
                preprocessed_commits = parallel(
                    delayed(preprocess_task)((ch, dot_git_dir))
                    for ch in commit_hash_chunk
                )

                # (2) store processed commits and update status
                for pc in preprocessed_commits:
                    _store_commit(pc, db)
                    pbar.update()
    else:
        # SEQUENTIAL version
        with tqdm.tqdm(
            total=total_number_of_commits,
            desc="Fetching commits (sequential)",
            bar_format=spojitr_utils.TQDM_FORMAT,
            disable=LOGGER.getEffectiveLevel() > logging.INFO,
        ) as pbar:
            for idx, commit_hash in enumerate(
                spojitr_utils.linewise(commit_hashes_file)
            ):
                preprocessed_commit = preprocess_task((commit_hash, dot_git_dir))
                _store_commit(preprocessed_commit, db)
                pbar.update()

    # remove temporary file
    commit_hashes_file.unlink()


def _fetch_jira_issues_since_issue_id(
    project_config: dict, since_issue_id=None
) -> typing.Tuple[typing.Iterable, int]:
    """
    :since_issue_id: optional issue_id, i.e. only issues with a greater id are fetched
                     if unspecified, all issues from the beginning are fetched
    :Returns: tuple containing (<stream of raw jira issues>, <number of expected results>)
    """
    if not since_issue_id:
        jql_query = f"project={project_config['jiraProjectKey']} ORDER BY id ASC"
    else:
        jql_query = f"project={project_config['jiraProjectKey']} AND id > \"{since_issue_id}\" ORDER BY id ASC"

    number_of_search_results = spojitr_utils.jira_get_number_of_search_results(
        jql_query, project_config["jiraRestUri"]
    )

    raw_issue_stream = spojitr_utils.jira_stream_jql_query_results(
        jql_query,
        project_config["jiraRestUri"],
        # issue_number_limit=100,
    )

    return raw_issue_stream, number_of_search_results


def _store_raw_jira_issue_stream(
    raw_issue_stream: typing.Iterable, number_of_issues: int, db: database.Database
):
    def xform_raw_jira_issues() -> typing.Iterable:
        """apply issue transformation chain
        """
        for raw_issue in raw_issue_stream:
            relevant_issue = spojitr_utils.extract_relevant_issue_data(raw_issue)

            desc = relevant_issue["description"]
            if desc is not None:
                relevant_issue["description"] = text_processing.preprocess(desc)

            yield relevant_issue

    batch_size = 100

    with tqdm.tqdm(
        total=number_of_issues,
        desc="Fetching issues",
        bar_format=spojitr_utils.TQDM_FORMAT,
        disable=LOGGER.getEffectiveLevel() > logging.INFO,
    ) as pbar:
        for issue_batch in spojitr_utils.grouper(batch_size, xform_raw_jira_issues()):
            db.insert_issues(issue_batch)
            pbar.update(len(issue_batch))


def _fetch_all_jira_issues(db: database.Database, project_config: dict):
    """Query all existing issues and store in local database

    :param db: spojitr database
    """
    LOGGER.info("Fetch jira issues ...")
    db.delete_table_contents("issues")

    raw_issue_stream, number_of_issues = _fetch_jira_issues_since_issue_id(
        project_config, since_issue_id=None
    )

    if number_of_issues > 0:
        _store_raw_jira_issue_stream(raw_issue_stream, number_of_issues, db)


def _update_jira_issues(db: database.Database, project_config: dict) -> list:
    """Query new issues, i.e. those no in database, and store them

    :Returns: list of new issue ids
    """

    def latest_issue_in_database() -> dict:
        stmt = """
            SELECT  issue_id, MAX(created_date) AS created_date
            FROM    issues
        """
        row = db.execute_sql(stmt).fetchone()
        return {"issue_id": row.issue_id, "created_date": row.created_date}

    LOGGER.info("Updating jira issues ...")

    # TODO FIXME: this only fetches true new issues, which is only half of the story
    #             technically we need to check, whether the state of existing
    #             issues in the database has changed (i.e. resolved state)
    #             -> in the worst case we have to rebuild the whole issue table
    #                which also enforces redoing similarity etc ...

    #             -> Possible solution: maybe only check the 'most recent' issues,
    #                i.e. 3 months ago, and the new ones?
    #                sqlite3: INSERT OR UPDATE might be usefull in this scenario
    latest_issue = latest_issue_in_database()
    LOGGER.debug("Latest issue: %s", latest_issue)

    latest_id = latest_issue["issue_id"]

    raw_issue_stream, number_of_issues = _fetch_jira_issues_since_issue_id(
        project_config, since_issue_id=latest_id
    )
    LOGGER.debug("Num new issues: %d", number_of_issues)

    if number_of_issues > 0:
        _store_raw_jira_issue_stream(raw_issue_stream, number_of_issues, db)

    # query new issue. We have to use the created_date and not issue_id, because
    # in sqlite3 FALCON-2434 IS NOT larger than FALCON-9!
    sql_stmt = f"""
        SELECT issue_id
        FROM issues
        WHERE created_date > "{latest_issue['created_date']}"
    """
    new_issues = [r.issue_id for r in db.execute_sql(sql_stmt)]
    LOGGER.info("New issue ids (%d): %s", len(new_issues), new_issues)
    return new_issues


def _fill_issues_to_change_set_table(db: database.Database):
    LOGGER.info("Establishing existing issue to commit links ...")
    db.delete_table_contents("issue_to_change_set")

    sql_stmt = """
        SELECT issue_id
        FROM issues
    """
    issue_ids = set(row.issue_id for row in db.execute_sql(sql_stmt))

    sql_stmt = """
        SELECT count (distinct commit_hash) AS total
        FROM change_set
    """
    total_commits = db.execute_sql(sql_stmt).fetchone().total

    sql_stmt = """
        SELECT distinct commit_hash, commit_msg
        FROM change_set
    """
    pairs = []
    with tqdm.tqdm(
        total=total_commits,
        desc="Linking",
        bar_format=spojitr_utils.TQDM_FORMAT,
        disable=LOGGER.getEffectiveLevel() > logging.INFO,
    ) as pbar:
        for commit_hash, commit_msg in db.execute_sql(sql_stmt):
            for issue_id in spojitr_utils.find_all_jira_identifiers(commit_msg):
                if issue_id in issue_ids:
                    pairs.append((issue_id, commit_hash))

            pbar.update()

    db.insert_issue_to_changeset_links(pairs)


def _get_commits_to_linked_issues(db: database.Database) -> dict:
    """
    :Returns: mapping from commit hash to list of linked issue ids
    """
    stmt = """
        SELECT commit_hash, issue_id
        FROM   issue_to_change_set
        ORDER BY commit_hash
    """

    commit_hash_to_issue_ids = collections.defaultdict(list)  # type: ignore
    for row in db.execute_sql(stmt):
        commit_hash_to_issue_ids[row.commit_hash].append(row.issue_id)

    return commit_hash_to_issue_ids


def _get_commit_messages(db: database.Database) -> dict:
    """
    :Returns: mapping from commit_hash to  message
    """
    stmt = """
        SELECT commit_hash, commit_msg
        FROM change_set
        WHERE commit_msg IS NOT NULL
    """
    return {r.commit_hash: r.commit_msg for r in db.execute_sql(stmt)}


def get_issue_descriptions(db: database.Database) -> dict:
    """
    :Returns: mapping from issue_id to description
    """
    stmt = """
        SELECT issue_id, issue_description
        FROM issues
        WHERE issue_description IS NOT NULL
    """
    return {r.issue_id: r.issue_description for r in db.execute_sql(stmt)}


def _iso_date_time_to_timestamp(d: str):
    if isinstance(d, str):
        t = datetime.datetime.strptime(d, "%Y-%m-%dT%H:%M:%Sz")
        return t.timestamp()
    return None


class IssueCandidateLookup:
    """filter issues based on commit, e.g. if a commit is performed after an issue is created,
    the issue is not relevant, if the commit is not performed between the creation and
    resolution + extra time for resolution of an issue (in case of closed issues),
    the issue is not relevant

    NOTE: this also could be done directly on the database, but profiling showed, that it is
          very slow

            SELECT  distinct issue_id, issue_description
            FROM    issues, change_set
            WHERE   commit_hash = "{hash}"
                        /* commit happens after issue creation */
                    AND (created_date <= committed_date)
                    AND (
                            /* issue is resolved and commit happens 'close to' resolved time */
                            (
                                (resolved_date IS NOT NULL)
                                AND
                                (committed_date <= datetime(resolved_date, "+{time} minutes"))
                            )
                            OR
                            /* issue is unresolved */
                            (resolved_date IS NULL)
                        )
    """

    def __init__(self, db: database.Database):
        sql_stmt = """
            SELECT issue_id, created_date, resolved_date
            FROM issues
            ORDER BY created_date
        """
        self._issues_ordered = [
            {
                "issue_id": r.issue_id,
                "created_ts": _iso_date_time_to_timestamp(r.created_date),
                "resolved_ts": _iso_date_time_to_timestamp(r.resolved_date),
            }
            for r in db.execute_sql(sql_stmt)
        ]

    def get_candidates_for_unlinked_commit(self, committed_date: str) -> set:
        """
        :param committed_date: date in ISO format
        :Returns: set of candidate issue_ids
        """
        # HACK: pure evil: this time base filtering exploits 'spojit' internals
        time_offset_seconds = 2160 * 60

        result = set()  # type: ignore
        commit_ts = _iso_date_time_to_timestamp(committed_date)

        for issue in self._issues_ordered:
            if commit_ts < issue["created_ts"]:
                # issue is newer than commit. And since issue are ordered by time,
                # all following are also younger and thus no candidates
                break

            if issue["resolved_ts"] is None:
                # issue is unresolved
                result.add(issue["issue_id"])
                continue

            if commit_ts <= issue["resolved_ts"] + time_offset_seconds:
                # issue is resolved and commit happens 'close to' resolved time
                result.add(issue["issue_id"])

        return result


def _get_all_commits(db: database.Database) -> typing.List[dict]:
    """
    :Returns: list of all commits as dictionaries, each dict representing a commit
                [{"author": "",
                  "author_email": "",
                  "commit_hash": "",
                  "committed_date": ""
                  "commit_msg": ""
                  "file_path": [ "file1", "file2", ... ]
                },
                ...]
    """

    def xform(row_proxy) -> dict:
        """transform row proxy to dict and separate the grouped file paths
        """
        commit = {k: row_proxy[k] for k in simple_column_names}
        commit["file_path"] = row_proxy["grouped_file_paths"].split("<SEP>")
        return commit

    # there is one row for each changed file per commit in the table.
    # -> let the db group multiple files together using separator <sep>
    simple_column_names = [
        "author",
        "author_email",
        "commit_hash",
        "committed_date",
        "commit_msg",
    ]
    sql_stmt = """
        SELECT {columns}, group_concat(file_path, "<SEP>") AS grouped_file_paths
        FROM change_set
        GROUP BY commit_hash
    """.format(
        columns=", ".join(simple_column_names)
    )

    return [xform(row) for row in db.execute_sql(sql_stmt)]


def _get_file_contents_by_commit(commit_hash: str, db: database.Database) -> dict:
    """
    :Returns: mapping from file name to file content
              {
                  "file1.java": <content>,
                  "path/file2.java": <content>,
                  ...
              }
    """
    sql_stmt = """
        SELECT file_path, file_content
        FROM change_set
        WHERE commit_hash="{hash}"
              AND file_content IS NOT NULL
    """.format(
        hash=commit_hash
    )

    return {r.file_path: r.file_content for r in db.execute_sql(sql_stmt)}


def _calculate_commit_to_issue_pairs_for_similarity(
    db: database.Database
) -> typing.List[dict]:
    """
    :Returns: list of structures. e.g.
                [
                    {
                        "commit_hash": "#adbc..",
                        "issue_ids" ["ISSUE-1", "ISSUE-2", ... ]
                        "reason": "linked"                        // justification
                    },
                    ...
                ]
    """
    # HACK: pure evil: apply internal spojit filter, so we only have to process
    # commits and changed files which are relevant
    all_commits = _get_all_commits(db)
    filtered_commits = spojit.profile._filter_change_sets(
        all_commits, spojit.artifact_filter.DefaultFilter()
    )

    commits_to_linked_issues = _get_commits_to_linked_issues(db)

    sql_stmt = """
        SELECT issue_id
        FROM issues
        WHERE issue_description IS NOT NULL
    """
    issue_ids_that_have_a_description = set(
        (r.issue_id for r in db.execute_sql(sql_stmt))
    )

    result: typing.List[dict] = []

    issue_canditate_lookup = IssueCandidateLookup(db)

    for commit in filtered_commits:
        commit_hash = commit["commit_hash"]
        if commit["commit_msg"] is None:
            # no message (unlikely) -> no similarity
            continue

        if commit_hash in commits_to_linked_issues:
            # commit is linked to issue(s) -> consider these issues for similarity
            linked_issue_ids = commits_to_linked_issues[commit_hash]
            issues_with_description = [
                issue_id
                for issue_id in linked_issue_ids
                if issue_id in issue_ids_that_have_a_description
            ]

            result.append(
                {
                    "commit_hash": commit_hash,
                    "issue_ids": issues_with_description,
                    "reason": "linked",
                }
            )
        else:
            # commit is not linked to issue(s) -> consider possible issues for similarity
            issue_candidates = issue_canditate_lookup.get_candidates_for_unlinked_commit(
                commit["committed_date"]
            )

            issues_with_description = [
                issue_id
                for issue_id in issue_candidates
                if issue_id in issue_ids_that_have_a_description
            ]

            result.append(
                {
                    "commit_hash": commit_hash,
                    "issue_ids": issues_with_description,
                    "reason": "candidate",
                }
            )

    return result


def _fill_issue_to_commit_similarity_table(db: database.Database):
    """
    """
    LOGGER.info("Calculate issue to commit similarity ...")
    db.delete_table_contents("issue_to_commit_similarity")

    commit_messages = _get_commit_messages(db)
    issue_descriptions = get_issue_descriptions(db)
    pairings = _calculate_commit_to_issue_pairs_for_similarity(db)

    issue_to_commit_similarity: typing.List[typing.Tuple] = []

    with tqdm.tqdm(
        total=len(pairings),
        desc="Calculate issue to commit similarity",
        bar_format=spojitr_utils.TQDM_FORMAT,
        disable=LOGGER.getEffectiveLevel() > logging.INFO,
    ) as pbar:
        for entry in pairings:
            commit_hash = entry["commit_hash"]
            issue_ids = entry["issue_ids"]

            pbar.update()

            if not issue_ids:
                continue

            query_doc = commit_messages[commit_hash]
            corpus_docs = [issue_descriptions[issue_id] for issue_id in issue_ids]
            similarities = text_processing.calculate_similarity(query_doc, corpus_docs)

            # store in result structure
            for issue_id, similarity in zip(issue_ids, similarities):
                issue_to_commit_similarity.append((issue_id, commit_hash, similarity))

    db.insert_issue_to_commit_similarities(issue_to_commit_similarity)


def _fill_issue_to_code_similarity_table(db: database.Database):
    def to_stable_lists(d: dict):
        """Split keys and values to two synchronized lists"""
        keys = []
        values = []
        for k, v in d.items():
            keys.append(k)
            values.append(v)
        return keys, values

    LOGGER.info("Calculate issue to source code similarity ...")
    db.delete_table_contents("issue_to_code_similarity")

    commit_messages = _get_commit_messages(db)
    issue_descriptions = get_issue_descriptions(db)
    pairings = _calculate_commit_to_issue_pairs_for_similarity(db)

    issue_to_code_similarity: typing.List[typing.Tuple] = []

    with tqdm.tqdm(
        total=len(pairings),
        desc="Calculate issue to source code similarity",
        bar_format=spojitr_utils.TQDM_FORMAT,
        disable=LOGGER.getEffectiveLevel() > logging.INFO,
    ) as pbar:
        for entry in pairings:
            commit_hash = entry["commit_hash"]
            issue_ids = entry["issue_ids"]

            pbar.update()

            if not issue_ids:
                continue

            # TODO: get contents by individual commit is super slow, as shown with profiling
            #
            #   Update: actually the slow performance only occurs if databse is on a docker volume share.
            #           In case it is stored in the container, it is much faster. however to proposed streaming
            #           solution is still faster
            #
            #             Faster solution:
            #               (1) order all commits by hash (in db) and stream them i.e. build generator in
            #                   python
            #               (2) check, whether in this case multi-processing with joblib on similarity further
            #                   boosts performance.
            #                   Right now it does not, because fetching from DB is the bottleneck
            file_content_mapping = _get_file_contents_by_commit(commit_hash, db)
            file_paths, file_contents = to_stable_lists(file_content_mapping)
            # issue ids need to be stable, too
            assert isinstance(issue_ids, list)

            # similarity for file contents of commit to each issue
            corpus_docs = zip(file_paths, file_contents)
            query_docs = [issue_descriptions[issue_id] for issue_id in issue_ids]
            simi_calculator = text_processing.CosineSimilarity(corpus_docs)

            # rows are the docs (files), columns are the queries (issue descriptions)
            similarity_matrix = simi_calculator.get_similarities(query_docs)
            assert similarity_matrix.shape == (len(file_paths), len(issue_ids))

            for col_idx, issue_id in enumerate(issue_ids):
                similarities = similarity_matrix[:, col_idx].flatten()
                # store in result structure
                for file_path, similarity in zip(file_paths, similarities):
                    issue_to_code_similarity.append(
                        (issue_id, commit_hash, file_path, similarity)
                    )

    db.insert_issue_to_code_similarities(issue_to_code_similarity)


def populate_database(
    project_config: dict, db_file: Path, spojitr_dir: Path, dot_git_dir: Path
):
    LOGGER.info("Build database ...")

    db = database.Database(db_file)
    db.create_tables()

    _fetch_all_commits(db, spojitr_dir=spojitr_dir, dot_git_dir=dot_git_dir)
    _fetch_all_jira_issues(db, project_config)
    _fill_issues_to_change_set_table(db)
    _fill_issue_to_commit_similarity_table(db)
    _fill_issue_to_code_similarity_table(db)


# *******************************************************************
# TESTS
# *******************************************************************


def test_fill_issues_to_change_set_table():
    db = database.Database("/data/spojitr_install/db.sqlite3")
    _fill_issues_to_change_set_table(db)


def test_fill_issue_to_commit_similarity_table():
    db = database.Database("/data/spojitr_install/db.sqlite3")
    _fill_issue_to_commit_similarity_table(db)


def test_fill_issue_to_code_similarity_table():
    db = database.Database("/data/spojitr_install/db.sqlite3")
    _fill_issue_to_code_similarity_table(db)


def test_calculate_commit_to_issue_pairs_for_similarity():
    db = database.Database("/data/spojitr_install/db.sqlite3")
    res = _calculate_commit_to_issue_pairs_for_similarity(db)
    LOGGER.info("res %s", res)


def test_populate_falcon():
    base: Path = Path("~/falcon").expanduser()
    dot_git_dir: Path = base / ".git"
    spojitr_dir: Path = base / ".spojitr"
    db_file = spojitr_dir / "falcon_perform.sqlite3"
    project_config = {
        "jiraProjectKey": "FALCON",
        "jiraRestUri": "https://issues.apache.org/jira/rest/api/2",
    }

    LOGGER.setLevel(logging.INFO)

    db = database.Database(db_file)
    db.create_tables()

    _fetch_all_commits(db, spojitr_dir=spojitr_dir, dot_git_dir=dot_git_dir)
    # _fetch_all_jira_issues(db, project_config)
    # _fill_issues_to_change_set_table(db)
    # _fill_issue_to_commit_similarity_table(db)
    # _fill_issue_to_code_similarity_table(db)


def test_update_jira_issues():
    base: Path = Path("~/falcon_checkout").expanduser()
    spojitr_dir: Path = base / ".spojitr"
    db_file = spojitr_dir / "falcon.sqlite3"
    project_config = {
        "jiraProjectKey": "FALCON",
        "jiraRestUri": "https://issues.apache.org/jira/rest/api/2",
    }

    db = database.Database(db_file)
    _update_jira_issues(db, project_config)


# *******************************************************************
# MAIN
# *******************************************************************


if __name__ == "__main__":
    logging.basicConfig(
        format="%(name)s %(asctime)s %(levelname)s %(message)s", level=logging.DEBUG
    )
    print(f"Hello from {__file__}")

    # test_fill_issues_to_change_set_table()
    # test_fill_issue_to_commit_similarity_table()
    # test_fill_issue_to_code_similarity_table()
    # test_calculate_commit_to_issue_pairs_for_similarity()
    test_populate_falcon()
    # test_update_jira_issues()
