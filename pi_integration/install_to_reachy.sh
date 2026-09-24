#!/usr/bin/env bash
# Simple script to install the custom Pi integration into an existing checkout
# of the official Reachy Mini conversation app.
#
# Usage:
#   ./install_to_reachy.sh /path/to/reachy_mini_conversation_app

set -euo pipefail

DEST="${1:-}"

if [[ -z "$DEST" ]]; then
  echo "Usage: $0 /path/to/reachy_mini_conversation_app"
  exit 1
fi

if [[ ! -d "$DEST/src" ]]; then
  echo "Error: Target directory does not look like the conversation app (no src/ directory)."
  exit 1
fi

echo "Copying custom Pi integration files to $DEST..."
cp -rv src/* "$DEST/src/"
cp -rv tests/* "$DEST/tests/"

echo "Done! Custom Reachy tools, text-turn server, and camera APIs are installed."
echo "Configure your environment variables in the conversation app and run it."
