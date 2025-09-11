#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <git-remote-url> [branch-name]"
  echo "Example: $0 git@github.com:YOURUSER/video-activity-logger.git main"
  exit 1
fi

REMOTE_URL="$1"
BRANCH="${2:-main}"

git init
git checkout -b "$BRANCH"
git add .
git commit -m "chore: initialize Video Activity Logger repo"
git remote add origin "$REMOTE_URL"
git push -u origin "$BRANCH"
echo "Pushed to $REMOTE_URL on branch $BRANCH"
