#!/bin/zsh
# Double-click to open kiri in the browser.
cd "$(dirname "$0")" || exit 1
command -v uv >/dev/null || { echo "需要先安裝 uv：https://docs.astral.sh/uv/"; read; exit 1; }
uv run kiri web
