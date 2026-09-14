#!/usr/bin/env bash
#
# One-time git initialisation for the ICL_Kinetics/Joe working folder.
#
# Run it once from this directory:
#     cd ~/Desktop/Postgraduate/ICL_Kinetics/Joe
#     bash git_setup.sh
#
# It stops before anything that needs your GitHub credentials -- it
# creates the local repo and the first commit, then prints the two
# commands you run yourself to attach a remote and push.
#
# Safe to abort at any point: nothing here touches paper/ (which is its
# own Overleaf-backed repo and is gitignored), and nothing is deleted.

set -euo pipefail

cd "$(dirname "$0")"

if [ -d .git ]; then
    echo "A git repo already exists here. Nothing to do."
    echo "Current status:"
    git status --short --branch
    exit 0
fi

echo "==> Initialising repository on branch 'main'"
git init --initial-branch=main

echo "==> Staging files (honouring .gitignore)"
git add -A

echo
echo "==> These files will be in the first commit:"
git status --short
echo
echo "    Total: $(git diff --cached --name-only | wc -l | tr -d ' ') files"
echo

# Sanity check: paper/ must not have been swept in. It's a nested repo
# with its own history and its own Overleaf remote -- committing it here
# would create a confusing second copy of the .tex that silently drifts
# out of sync with Overleaf.
if git diff --cached --name-only | grep -q '^paper/'; then
    echo "WARNING: files under paper/ are staged, which is not intended."
    echo "         Check .gitignore before committing. Aborting."
    exit 1
fi

echo "==> Creating the initial commit"
git commit -q -m "Initial commit: barrel pyrolysis 1D solver and training code

Coupled stochastic-mixture heat conduction with two-group neutron
diffusion and point kinetics for a PuO2 + combustible-waste storage
drum, including the moderator-percolation submodel (percolation.py)
and its per-cell dynamic correlation length.

The LaTeX source under paper/ is tracked separately in its own
Overleaf-backed repository and is excluded here by .gitignore."

echo
echo "Done. Local repo created with $(git rev-list --count HEAD) commit."
echo
echo "─────────────────────────────────────────────────────────────────"
echo "Next, run these two yourself (they need your GitHub account):"
echo
echo "  1. Create an empty repo on github.com -- no README, no"
echo "     .gitignore, no licence, or step 2 will complain about"
echo "     unrelated histories."
echo
echo "  2. Attach it and push:"
echo
echo "       git remote add origin git@github.com:<you>/<repo>.git"
echo "       git push -u origin main"
echo
echo "     Use the https:// URL instead if you haven't set up SSH keys."
echo "─────────────────────────────────────────────────────────────────"
