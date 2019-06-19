#!/usr/bin/env sh

#####################################################################
# CONFIGURATION

COMMIT_HASH=d5e40e3393b4fb1e2f3c60d158191ec3e81302f8
COMMIT_MESSAGE="Enable numReducers option for Distinct operations."
CORRECT_ISSUE_ID="CRUNCH-642"

#####################################################################
# run

this_dir="$(dirname "$0")"
$this_dir/_run_demo.sh ${COMMIT_HASH} "${COMMIT_MESSAGE}" "${CORRECT_ISSUE_ID}"