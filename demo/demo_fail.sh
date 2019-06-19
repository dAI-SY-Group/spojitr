#!/usr/bin/env sh

#####################################################################
# CONFIGURATION

COMMIT_HASH=571b90c03e3010e7bb9badf4e6e441ab2164be56
COMMIT_MESSAGE="Avoid unnecessary last modified time retrieval"
CORRECT_ISSUE_ID="CRUNCH-678"

#####################################################################
# run

this_dir="$(dirname "$0")"
$this_dir/_run_demo.sh ${COMMIT_HASH} "${COMMIT_MESSAGE}" "${CORRECT_ISSUE_ID}"
