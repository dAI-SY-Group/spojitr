import logging
import sqlalchemy as sqa
import typing


LOGGER = logging.getLogger(__name__)

#####################################################################
# TABLES
#####################################################################

CHANGE_SET_TBL_STMT = """
    CREATE TABLE IF NOT EXISTS change_set (
        commit_hash text NOT NULL,
        author text NOT NULL,
        author_email text NOT NULL,
        committed_date text NOT NULL,
        commit_msg text NOT NULL,
        file_path text,
        file_content text
    );"""


ISSUES_TBL_STMT = """
    CREATE TABLE IF NOT EXISTS issues (
        issue_id text NOT NULL,
        assignee text,
        assignee_username text,
        status text,
        priority text,
        created_date text NOT NULL,
        type text NOT NULL,
        resolved_date text,
        resolution text,
        issue_description text,
        issue_summary text
    );"""

ISSUE_TO_CHANGE_SET_TBL_STATEMENT = """
    CREATE TABLE IF NOT EXISTS issue_to_change_set (
        commit_hash text NOT NULL,
        issue_id text NOT NULL
    );"""


ISSUE_TO_COMMIT_SIMILARITY_TBL_STMT = """
    CREATE TABLE IF NOT EXISTS issue_to_commit_similarity (
        issue_id text NOT NULL,
        commit_hash text NOT NULL,
        issue_to_commit_sim real,
        UNIQUE(issue_id, commit_hash)
    );"""


ISSUE_TO_CODE_SIMILARITY_TBL_STMT = """
    CREATE TABLE IF NOT EXISTS issue_to_code_similarity (
        issue_id text NOT NULL,
        commit_hash text NOT NULL,
        file_path text NOT NULL,
        issue_to_code_sim real,
        UNIQUE(issue_id, commit_hash, file_path)
    );"""


#####################################################################
# DATABASE
#####################################################################


class Database:
    def __init__(self, database_file_name):
        url = "sqlite:///%s" % database_file_name
        self._engine = sqa.create_engine(url)

    def execute_sql(self, sql):
        """Execute raw sql statement
        :param sql: sql statement
        :return: resulting rows
        """
        return self._engine.execute(sql)

    def _drop_all_tables(self):
        LOGGER.debug("drop all tables")
        table_names = [
            "change_set",
            "issues",
            "issue_to_change_set",
            "issue_to_commit_similarity",
            "issue_to_code_similarity",
        ]

        for tbl in table_names:
            stmt = f"DROP TABLE IF EXISTS {tbl};"
            self.execute_sql(stmt)

    def delete_table_contents(self, tbl_name: str):
        self.execute_sql(f"DELETE FROM {tbl_name}")

    def create_tables(self):
        LOGGER.debug("create tables")
        stmts = [
            CHANGE_SET_TBL_STMT,
            ISSUES_TBL_STMT,
            ISSUE_TO_CHANGE_SET_TBL_STATEMENT,
            ISSUE_TO_COMMIT_SIMILARITY_TBL_STMT,
            ISSUE_TO_CODE_SIMILARITY_TBL_STMT,
        ]

        for stmt in stmts:
            self.execute_sql(stmt)

    def insert_change_set_entry(
        self,
        commit_hash: str,
        author: str,
        author_email: str,
        committed_date: str,
        commit_msg: str,
        file_infos: typing.List[typing.Tuple],
    ):
        stmt = """INSERT INTO change_set(
            commit_hash, author, author_email, committed_date, commit_msg, file_path, file_content)
            VALUES(?,?,?,?,?,?,?)"""

        with self._engine.connect() as connection:
            trans = connection.begin()
            for file_path, file_content in file_infos:
                params = [
                    commit_hash,
                    author,
                    author_email,
                    committed_date,
                    commit_msg,
                    file_path,
                    file_content,
                ]
                connection.execute(stmt, *params)
            trans.commit()

    def insert_issues(self, issues: typing.Iterable[dict]):
        def to_params(issue: dict) -> list:
            return [
                issue["issue_id"],
                issue["assignee"],
                issue["assignee_username"],
                issue["status"],
                issue["priority"],
                issue["created_date"],
                issue["type"],
                issue["resolved_date"],
                issue["resolution"],
                issue["description"],
                issue["summary"],
            ]

        stmt = """INSERT INTO issues(
            issue_id, assignee, assignee_username, status, priority,
            created_date, type, resolved_date, resolution, issue_description, issue_summary)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)"""

        with self._engine.connect() as connection:
            trans = connection.begin()
            for issue in issues:
                connection.execute(stmt, *to_params(issue))
            trans.commit()

    def insert_issue_to_changeset_links(self, links: typing.Iterable[typing.Tuple]):
        """
        :param links: list of tuples (issue_id, commit_hash)
        """
        stmt = """INSERT INTO issue_to_change_set(
            issue_id, commit_hash)
            VALUES(?, ?)"""

        with self._engine.connect() as connection:
            trans = connection.begin()
            for issue_id, commit_hash in links:
                connection.execute(stmt, issue_id, commit_hash)
            trans.commit()

    def insert_issue_to_commit_similarities(self, similarities: typing.Iterable):
        """
        :param similarities: list of tuples (issue_id, commit_hash, similarity)
        """
        stmt = """INSERT INTO issue_to_commit_similarity(
            issue_id, commit_hash, issue_to_commit_sim)
            VALUES(?, ?, ?)"""

        with self._engine.connect() as connection:
            trans = connection.begin()
            for issue_id, commit_hash, value in similarities:
                connection.execute(stmt, issue_id, commit_hash, value)
            trans.commit()

    def insert_issue_to_code_similarities(self, similarities: typing.Iterable):
        """
        :param similarities: list of tuples (issue_id, commit_hash, file_path, similarity)
        """
        stmt = """INSERT INTO issue_to_code_similarity(
            issue_id, commit_hash, file_path, issue_to_code_sim)
            VALUES(?, ?, ?, ?)"""

        with self._engine.connect() as connection:
            trans = connection.begin()
            for issue_id, commit_hash, file_path, value in similarities:
                connection.execute(stmt, issue_id, commit_hash, file_path, value)
            trans.commit()


# *******************************************************************
# TESTS
# *******************************************************************


def test_create_database():
    db = Database("/data/spojitr_install/dummy_db.sqlite3")
    db.create_tables()


# *******************************************************************
# MAIN
# *******************************************************************


if __name__ == "__main__":
    print(f"Hello from {__file__}")
    test_create_database()
