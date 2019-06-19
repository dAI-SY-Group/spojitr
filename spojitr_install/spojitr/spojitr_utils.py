"""
Utility functions used by the scripts in spojitr installation
"""

import copy
import datetime
import itertools
import json
import logging
import pprint
import re
import requests
import subprocess
import time
import tqdm
import typing
import urllib


from pathlib import Path

# *******************************************************************
# CONFIGURATION
# *******************************************************************


LOGGER = logging.getLogger(__file__)

# see: https://bit.ly/2L2iZHg
# \b is word boundary
JIRA_IDENTIFIER_REGEX = re.compile(r"\b[A-Z]+-\d+\b")

TQDM_FORMAT = "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"

# *******************************************************************
# FUNCTIONS
# *******************************************************************


def find_all_jira_identifiers(text: str) -> typing.Set[str]:
    # HACK: the preprocessing texts, which are stored in the db, have all been lowercased
    #       but jira keys are uppercased so we reversed this procedure in a hacky way!
    text_upper = text.upper()
    return set(re.findall(JIRA_IDENTIFIER_REGEX, text_upper))


def capture_command_output(args: list) -> str:
    # LOGGER.debug("capture command output: {}".format(" ".join(args)))
    result = subprocess.check_output(args, shell=False)
    return result.decode("utf-8", errors="ignore")


def grouper(n, iterable: typing.Iterable) -> typing.Iterable:
    """Collect data into fixed-length chunks or blocks of size 'n'

    The last chunk may be shorter, i.e. there is no filling up!

    >>> grouper(3, range(7))
    [0, 1, 2], [3, 4, 5], [6]

    :see: https://stackoverflow.com/a/8991553
    """
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def linewise(file_path):
    with open(file_path, "r") as fp:
        for line in fp:
            yield line.strip()


def _get_git_modified_text_files(
    commit_hash: str, dot_git_dir: Path
) -> typing.Iterable:
    def extract_columns(line: str, col_separator: str) -> dict:
        # LOGGER.debug("%s -> columns %s", line, line.split("\t"))
        added, deleted, path = line.split(col_separator)
        return {"add": added, "del": deleted, "path": path}

    def is_text_file(stats: dict) -> bool:
        # binary files are marked with "-" for added/deleted lines in numstat
        return stats["add"] != "-"

    def parse_numstat(text):
        """git numstat output looks like this

            10  11  foo.txt
            0   1   bar.txt

        columns (separated by tabs): added lines, deleted lines, paths
        """
        return [extract_columns(l, col_separator="\t") for l in text.split("\n") if l]

    common_args = ["git", f"--git-dir={dot_git_dir}"]

    # about filtering
    # - we exclude files that were deleted, because we cannot get their content any more
    # - we exclude files that were rename, because --numstat output (needed for binary detection)
    #   is weird, e.g. {prism => common}/src/main/java/org/apache/falcon/security/HostnameFilter.java
    # TODO: are we losing information this way?
    args = common_args + [
        "show",
        "--diff-filter=dr",  # ignore (d)elete,(r)enamed files
        "--pretty=format:",
        "--numstat",
        commit_hash,
    ]

    numstat_output = capture_command_output(args)
    # LOGGER.info("raw output %s", numstat_output)
    file_stats = parse_numstat(numstat_output)
    # LOGGER.debug("file stats %s", file_stats)

    text_file_stats = filter(is_text_file, file_stats)
    # LOGGER.debug("text stats %s", list(text_file_stats))

    return text_file_stats


# TODO FIXME: change 'cn'/'ce' to 'an'/'ae' to respect amending?
#   - the 'c' arguments are about the committer, i.e. the one who initially committed the path
#   - the 'a' arguments are about the author, i.e. the one who modified the commit the last time,
#     e.g. by amending the message
COMMIT_INFO_MAPPING = [
    ("%cn", "author"),
    ("%ce", "email"),
    ("%ci", "date"),
    ("%s", "msg"),
]
COMMIT_PRETTY_SEPARATOR = "<XXXSEPXXX>"
COMMIT_INFO_PRETTY_FORMAT = COMMIT_PRETTY_SEPARATOR.join(
    [v[0] for v in COMMIT_INFO_MAPPING]
)


def _format_git_timestamp(git_timestamp: str) -> str:
    return datetime.datetime.strptime(git_timestamp, "%Y-%m-%d %H:%M:%S %z").strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _git_get_commit_meta(commit_hash: str, dot_git_dir: Path) -> dict:
    """retrieve author, email etc from commit
    """
    args = [
        "git",
        f"--git-dir={dot_git_dir}",
        "log",
        f"--pretty=format:{COMMIT_INFO_PRETTY_FORMAT}",
        "-n",
        "1",
        commit_hash,
    ]

    commit_info = capture_command_output(args)
    field_values = commit_info.split(COMMIT_PRETTY_SEPARATOR)
    assert len(field_values) == len(COMMIT_INFO_MAPPING)

    field_names = (v[1] for v in COMMIT_INFO_MAPPING)
    result = {k: v for k, v in zip(field_names, field_values)}

    # fix the date
    result["date"] = _format_git_timestamp(result["date"])
    return result


def git_get_latest_commit_hash(dot_git_dir: Path) -> str:
    args = ["git", f"--git-dir={dot_git_dir}", "log", "--pretty=format:%H", "-n", "1"]
    commit_hash = capture_command_output(args)
    return commit_hash.strip()


def git_get_commit_information(commit_hash: str, dot_git_dir: Path) -> dict:
    """retrieve spojitr relevant data from a given commit

    Returns: dictionary describing the commit

        Example:

        {
            "author": "John Doe",
            "email": "jdoe@cyborg.org",
            "date": "2018-12-24 13:23:26 +0000",
            "msg": "this is the commit message",
            "paths": ["text_file1.txt", "/path/text_file2.txt"]
        }
    """
    # TODO: The returned structure (i.e. field names) differs from the
    #       ones used in the database as wel las by spojit datasource
    #       This requires multiple 'reformats' and introduces confusion
    #       -> try to unify the formats and stick to one, at least within
    #       spojitr, since we can't modify spojit library

    LOGGER.debug("hash %s", commit_hash)

    # fetch all modified text files
    text_file_stats = list(_get_git_modified_text_files(commit_hash, dot_git_dir))
    LOGGER.debug("text stats %s", text_file_stats)

    # get commit information
    commit_info = _git_get_commit_meta(commit_hash, dot_git_dir)
    LOGGER.debug("Commit info: %s", commit_info)

    result = copy.deepcopy(commit_info)
    result["paths"] = [e["path"] for e in text_file_stats]

    return result


def git_get_file_content(commit_hash: str, file_path: Path, dot_git_dir: Path) -> str:
    args = ["git", f"--git-dir={dot_git_dir}", "show", f"{commit_hash}:{file_path}"]
    return capture_command_output(args)


def git_amend_last_commit(message: str, dot_git_dir: Path, author: str = None):
    """
    :param author: changed author information (name and email), which requires the format
                "Homer Simpson <homer@springfield.com>"
    """

    args = ["git", f"--git-dir={dot_git_dir}", "commit", "--amend", "-m", message]

    if author:
        args += ["--author", author]

    amend_output = capture_command_output(args)
    return amend_output


def extract_relevant_issue_data(raw_jira_issue: dict) -> dict:
    """Extract spojitr relevant data from raw REST response
    """

    def get_field(field_name):
        entry = raw_jira_issue["fields"][field_name]
        if isinstance(entry, dict):
            return entry["name"]

        return entry

    def convert_jira_time(jira_time: str) -> str:
        return datetime.datetime.strptime(jira_time, "%Y-%m-%dT%H:%M:%S.%f%z").strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    # LOGGER.debug(data)

    issue = {}

    issue["issue_id"] = raw_jira_issue["key"]
    issue["type"] = get_field("issuetype")
    issue["resolution"] = get_field("resolution")
    issue["priority"] = get_field("priority")
    issue["status"] = get_field("status")
    issue["created_date"] = convert_jira_time(get_field("created"))
    issue["summary"] = get_field("summary")

    assignee = raw_jira_issue["fields"]["assignee"]
    if isinstance(assignee, dict):
        issue["assignee_username"] = assignee["name"]
        issue["assignee"] = assignee["displayName"]
    else:
        issue["assignee_username"] = assignee
        issue["assignee"] = assignee

    resolution_date = raw_jira_issue["fields"]["resolutiondate"]
    if resolution_date is not None:
        issue["resolved_date"] = convert_jira_time(resolution_date)
    else:
        issue["resolved_date"] = None

    description = raw_jira_issue["fields"]["description"]
    if description is not None:
        issue["description"] = description
    else:
        issue["description"] = None

    return issue


def jira_get_number_of_search_results(jql_query: str, jira_rest_uri: str) -> int:
    """
    :param jql_query: Query in jira query language (JQL)
    :param jira_rest_uri, e.g. "https://issues.apache.org/jira/rest/api/2"
    """
    params = {"jql": urllib.parse.quote(jql_query), "maxResults": 0}

    url = f"{jira_rest_uri}/search"
    params_str = "&".join(f"{k}={v}" for k, v in params.items())

    # TODO: error handling
    response = requests.get(url, params=params_str)
    data = response.json()

    return data["total"]


def _jira_query_issues_paginated(
    jql_query: str, jira_rest_uri: str, start_at: int = 0, page_size: int = 100
) -> list:
    """Query jira issues with JQL and pagination support

    :param jql_query: Query in jira query language (JQL), e.g.
                        "project=MNG ORDER BY id ASC"
    :Returns: List of matching issues in json format
    """

    LOGGER.debug(
        "query jira with %s paginated startAt=%d, maxResults=%d",
        jql_query,
        start_at,
        page_size,
    )

    params = {
        "jql": urllib.parse.quote(jql_query),
        "startAt": start_at,
        "maxResults": page_size,
    }

    url = f"{jira_rest_uri}/search"
    params_str = "&".join(f"{k}={v}" for k, v in params.items())

    # TODO: error handling
    response = requests.get(url, params=params_str)
    data = response.json()
    return data["issues"]


def jira_stream_jql_query_results(
    jql_query: str, jira_rest_uri: str, issue_number_limit=None
) -> typing.Iterable:
    """Stream all jira issues of a JQL query

    :param issue_number_limit: upper limit of number of issues to receive

    "returns: stream of raw jira issues
    """
    max_number_of_issues = -1
    if isinstance(issue_number_limit, int):
        max_number_of_issues = issue_number_limit
        LOGGER.debug("Set max results to %d", max_number_of_issues)

    start_at = 0
    page_size = 500
    num_received_issues = 0

    while True:
        issues_chunk = _jira_query_issues_paginated(
            jql_query, jira_rest_uri, start_at, page_size
        )

        chunk_size = len(issues_chunk)
        LOGGER.debug("Received issue chunk of size %d", chunk_size)

        if chunk_size > 0:
            num_received_issues += chunk_size

            # stream the results
            yield from issues_chunk

            # check early exit
            if (max_number_of_issues >= 0) and (
                num_received_issues >= max_number_of_issues
            ):
                LOGGER.warning(
                    "EARLY EXIT. Received %d issues which exceeds requested limit of %d.",
                    num_received_issues,
                    max_number_of_issues,
                )
                break

            start_at += page_size

            # throttle ?
            if (num_received_issues > 0) and (num_received_issues % 4000 == 0):
                wait_time_sec = 120
                LOGGER.info(
                    "Throttling Jira API requests. Waiting for %d seconds ...",
                    wait_time_sec,
                )
                time.sleep(wait_time_sec)

            continue

        LOGGER.debug("Received %d issues. Finish", num_received_issues)
        break


class Progress:
    def __init__(self, callback, report_n_steps=100, max_steps=None):
        self._callback = callback
        self._report_n_steps = report_n_steps
        self._max_steps = max_steps
        self._steps = 0

    def update(self, steps=1, data=None):
        self._steps += steps
        if (self._steps > 0) and (self._steps % self._report_n_steps == 0):
            self._callback(self._steps, self._max_steps, data)

    @property
    def current_step(self) -> int:
        return self._steps


# *******************************************************************
# TESTS
# *******************************************************************


def test_grouper():
    for i, chunk in enumerate(grouper(3, range(7))):
        LOGGER.info("Chunk %d: %s", i, list(chunk))


def test_jira_stream_jql_query_results():
    query = f"project=MNG ORDER BY id ASC"

    issue_stream = jira_stream_jql_query_results(
        query, "https://issues.apache.org/jira/rest/api/2", issue_number_limit=17
    )

    data = {"issues": list(issue_stream)}

    with open("/data/spojitr_install/test_data/jira_new.json", "w") as fp:
        json.dump(data, fp)

    # with open("/data/spojitr_install/test_data/jira.json", "r") as fp:
    #     data = json.load(fp)
    #     for issue in data["issues"]:
    #         issue = _extract_relevant_issue_data(issue)
    #         LOGGER.debug(issue)
    # pass


def test_get_jira_issues_paginated():
    issues = query_jira_issues_paginated(
        "MNG", "https://issues.apache.org/jira/rest/api/2", start_at=5388, page_size=100
    )

    LOGGER.info("number of issues: %s", len(issues))


def test_jira_requests():
    num_issues = jira_get_number_of_search_results(
        "MNG", "https://issues.apache.org/jira/rest/api/2"
    )
    LOGGER.info("Number of issues: %d", num_issues)


def test_find_all_jira_identifiers():
    texts = [
        "normal sentence",
        "MAVEN-10 a couple of ids FOO-123 and more BAR-3",
        "FOO-10. This has duplicates. FOO-10, Yes!",
        "Another Bar-11 boing",
    ]

    for t in texts:
        LOGGER.debug("text   : %s\nids    : %s", t, find_all_jira_identifiers(t))


def test_progress():
    p = Progress(lambda x, y, d: LOGGER.info("   every 3 Step %d/%s", x, y), 3)
    for i in range(10):
        LOGGER.info("raw %d", i)
        p.update()

    LOGGER.info("-" * 10)

    upper_bound = 13
    p = Progress(
        lambda x, y, d: LOGGER.info("   every 2 Step %d/%d", x, y - 1), 2, upper_bound
    )
    for i in range(upper_bound):
        LOGGER.info("raw %d", i)
        p.update()


def test_progress2():
    with tqdm.tqdm(
        total=200,
        desc="Processing commits",
        bar_format=TQDM_FORMAT,
        disable=LOGGER.getEffectiveLevel() > logging.INFO,
    ) as pbar:
        for i in range(20):
            time.sleep(0.1)
            pbar.update(10)


def test_git_amend_last_commit():
    p = Path("~/example_repo/.git").expanduser()
    s = git_amend_last_commit("This is an amended message!", p)
    LOGGER.info("output: %s", s)


def test_git_amend_last_commit_workflow():
    p = Path("~/example_repo/.git").expanduser()
    commit_hash = git_get_latest_commit_hash(p)
    commit_info = git_get_commit_information(commit_hash, p)
    LOGGER.info("Latest commit %s:\n%s", commit_hash, pprint.pformat(commit_info))
    new_msg = "Amended: " + commit_info["msg"]
    LOGGER.info("Change message to: %s", new_msg)

    author = "tony stark <tony@stark-industries.com>"
    git_amend_last_commit(new_msg, p, author_name=author)

    commit_hash = git_get_latest_commit_hash(p)
    commit_info = git_get_commit_information(commit_hash, p)
    LOGGER.info("NEW latest commit %s\n%s", commit_hash, pprint.pformat(commit_info))


# *******************************************************************
# MAIN
# *******************************************************************


if __name__ == "__main__":
    logging.basicConfig(format="%(name)s %(levelname)s %(message)s", level=logging.INFO)
    print(f"Hello from {__file__}")

    # test_grouper()
    # test_jira_stream_jql_query_results()
    # test_jira_requests()
    # test_get_jira_issues_paginated()
    # test_find_all_jira_identifiers()
    # test_progress()
    # test_progress2()
    # test_git_amend_last_commit()
    test_git_amend_last_commit_workflow()
