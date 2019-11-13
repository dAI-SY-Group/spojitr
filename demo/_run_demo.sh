#!/usr/bin/env sh

#####################################################################
# read commandline arguments

COMMIT_HASH=$1
COMMIT_MESSAGE=$2
CORRECT_ISSUE_ID=$3

#####################################################################
# describe what happens

cat << EOF
---------------------------------------------------------------------
* Setting up demo for commit ${COMMIT_HASH}
* The given commit message was

    ${COMMIT_MESSAGE}

* The commit was linked to issue id ${CORRECT_ISSUE_ID}
---------------------------------------------------------------------
EOF

#####################################################################
# inform spojitr about demo, i.e. to use different model etc

cat << EOF > /root/crunch/.spojitr/demo.json
{
    "commit_hash": "${COMMIT_HASH}"
}
EOF

#####################################################################
# create a git branch and recreate the situation at time of commit

PATCH_FILE=/root/crunch/.spojitr/patch_${COMMIT_HASH}.diff

# the changes made in the commit
git diff ${COMMIT_HASH}^1..${COMMIT_HASH} > ${PATCH_FILE}

# create branch pointing _before_ commit and apply the patch
git checkout master
git branch -D demo_branch
git checkout -b demo_branch ${COMMIT_HASH}^1
git apply ${PATCH_FILE}
