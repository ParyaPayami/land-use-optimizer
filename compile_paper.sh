#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "============================================="
echo "PIMALUOS Manuscript Compilation Script"
echo "============================================="

# Explicitly add standard macOS MacTeX / TeX Live locations to the PATH
export PATH="/Library/TeX/texbin:/usr/local/texlive/2026/bin/universal-darwin:/usr/local/texlive/2025/bin/universal-darwin:/usr/local/texlive/2024/bin/universal-darwin:/opt/homebrew/bin:/usr/local/bin:$PATH"

# Check if pdflatex is found
if ! command -v pdflatex &> /dev/null; then
    echo "❌ ERROR: pdflatex could not be found."
    echo "It seems MacTeX is installed but its CLI tools are not yet linked."
    echo "Please try running this command to re-initialize your paths:"
    echo "  eval \"\$(/usr/libexec/path_helper)\""
    echo "Or verify MacTeX is fully installed on your system."
    exit 1
fi

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Ensure all figures are generated from the checkpoint
if [ -d ".venv" ]; then
    echo "Generating latest figures from cache/checkpoint..."
    ./.venv/bin/python scripts/generate_all_figures.py
fi

echo "Step 1: Compiling self-contained FINAL_SUBMISSION.tex..."
pdflatex -interaction=nonstopmode FINAL_SUBMISSION.tex
bibtex FINAL_SUBMISSION
pdflatex -interaction=nonstopmode FINAL_SUBMISSION.tex
pdflatex -interaction=nonstopmode FINAL_SUBMISSION.tex

echo "Step 2: Copying compiled PDF to root as PIMALUOS.pdf..."
cp FINAL_SUBMISSION.pdf PIMALUOS.pdf

echo "Step 3: Compiling modular manuscript (manuscript tex/paper.tex)..."
mkdir -p "manuscript tex/results"
cp -r results/figures "manuscript tex/results/"
cd "manuscript tex"
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex

echo "Step 4: Copying modular PDF to manuscript folder..."
cp paper.pdf ../PIMALUOS_modular.pdf

echo "============================================="
echo "Success! Compiled papers generated:"
echo "  1. Root: PIMALUOS.pdf (Self-contained, complete)"
echo "  2. Root: PIMALUOS_modular.pdf (Modular build)"
echo "============================================="
