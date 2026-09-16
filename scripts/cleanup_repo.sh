#!/usr/bin/env bash
# Removes stray files and duplicate uploads accumulated during development.
# Review the list it prints BEFORE confirming - this deletes files.
#
# Run from the project root:
#   bash scripts/cleanup_repo.sh

set -euo pipefail

echo "== stray files =="
STRAY=(notes.txt notebooks/noemptydirectory)
for f in "${STRAY[@]}"; do
    [ -e "$f" ] && echo "  $f"
done

echo ""
echo "== duplicate uploads in models/ (pattern: 'name (N).ext') =="
find models -maxdepth 1 -regextype posix-extended -regex '.*\([0-9]+\)\.[a-z]+' -print

echo ""
echo "== unexplained files in models/ (no recognized extension) =="
find models -maxdepth 1 -type f ! -name "*.h5" ! -name "*.joblib" ! -name "*.pkl" \
    ! -name "*.json" ! -name "*.csv" ! -name "*.html" ! -name ".gitkeep" -print

echo ""
read -p "Delete all of the above? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted - nothing deleted."
    exit 0
fi

for f in "${STRAY[@]}"; do
    [ -e "$f" ] && rm -v "$f"
done

find models -maxdepth 1 -regextype posix-extended -regex '.*\([0-9]+\)\.[a-z]+' -delete -print

echo "Done. Review 'unexplained files' above manually - not deleted automatically."
echo "Next: git add -A && git commit -m 'chore: remove stray and duplicate files'"
