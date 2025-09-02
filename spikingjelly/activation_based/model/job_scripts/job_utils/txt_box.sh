#!/usr/bin/env bash
# txt_box.sh — wrap text into a rectangular box
# Usage:
#   ./txt_box.sh -w 50 "Paragraphs and\nmultiple lines..."
#   echo -e "para1\n\npara2\nline2 of para2" | ./txt_box.sh -w 40

set -euo pipefail

# Default parameters
# -w: width of the box
# -c: character used for the border
WIDTH=60
BORDER_CHAR="#"

# Parse options, override defaults when received external arguments
while getopts ":w:c:" opt; do
  case "$opt" in
    w) WIDTH="$OPTARG" ;;
    c) BORDER_CHAR="$OPTARG" ;;
    *) echo "Usage: $0 [-w width] [-c border_char] [text...]"; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

# Read input (args or stdin) — keep line breaks
if [ $# -gt 0 ]; then
  INPUT="$*"
else
  INPUT="$(cat)"
fi

# Prepare wrapped lines array
LINES=()
while IFS= read -r line; do
  if [ -z "$line" ]; then
    # Preserve blank lines
    LINES+=("")
  else
    # Fold long lines at word boundaries
    while IFS= read -r wrapped; do
      LINES+=("$wrapped")
    done < <(printf "%s\n" "$line" | fold -s -w "$WIDTH")
  fi
done <<< "$INPUT"

# Handle case: no lines at all
if [ ${#LINES[@]} -eq 0 ]; then
  LINES=(" ")
fi

# Find max width of wrapped content
MAX=0
for line in "${LINES[@]}"; do
  (( ${#line} > MAX )) && MAX=${#line}
done

# Build border
BORDER_LEN=$((MAX + 4))
BORDER=$(printf '%*s' "$BORDER_LEN" '' | tr ' ' "$BORDER_CHAR")

# Print box
echo "$BORDER"
for line in "${LINES[@]}"; do
  printf "%s %-*s %s\n" "$BORDER_CHAR" "$MAX" "$line" "$BORDER_CHAR"
done
echo "$BORDER"
