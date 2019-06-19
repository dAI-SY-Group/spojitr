#!/usr/bin/env sh

#####################################################################
# CONFIGURATION

COMMIT_HASH=869aac60c9d3b5bef10b4e907ec3840be2d8c20e
COMMIT_MESSAGE="Fix .equals and .hashCode for Targets"
CORRECT_ISSUE_ID="CRUNCH-684"

#####################################################################
# run

this_dir="$(dirname "$0")"
$this_dir/_run_demo.sh ${COMMIT_HASH} "${COMMIT_MESSAGE}" "${CORRECT_ISSUE_ID}"
