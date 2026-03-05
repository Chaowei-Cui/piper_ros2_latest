#!/usr/bin/env bash
cd collect_data
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/view_data.py"
DATASET_DIR="/home/agilex/data/sss"
TASK_NAME="123"
EPISODE_IDX=26
FILE_PATH=""
FPS=30
START=0
SHOW_DEPTH=0
SHOW_VELOCITY=0
SHOW_EFFORT=0

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --task <task_name> [options]
  $(basename "$0") --file <episode.hdf5> [options]

Options:
  --task <name>         Task folder under dataset_dir
  --episode <idx>       Episode index (default: 0)
  --dataset-dir <dir>   Dataset root (default: /home/agilex/data)
  --file <path>         Direct path to hdf5 file
  --fps <num>           Playback fps (default: 20)
  --start <idx>         Start frame (default: 0)
  --depth               Show depth images
  --vel                 Show velocity text
  --effort              Show effort text
  -h, --help            Show help

Examples:
  $(basename "$0") --task bin_packing_ljm_all_up_random_screwdriver_pan_hand_cream_lipstick --episode 0
  $(basename "$0") --file /home/agilex/data/xxx/episode_0.hdf5 --depth --vel
EOF
}

while [[ $# -gt 0 ]]; do
  case "$01" in
    --task)
      TASK_NAME="$2"
      shift 2
      ;;
    --episode)
      EPISODE_IDX="$2"
      shift 2
      ;;
    --dataset-dir)
      DATASET_DIR="$2"
      shift 2
      ;;
    --file)
      FILE_PATH="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --start)
      START="$2"
      shift 2
      ;;
    --depth)
      SHOW_DEPTH=1
      shift
      ;;
    --vel)
      SHOW_VELOCITY=1
      shift
      ;;
    --effort)
      SHOW_EFFORT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "Cannot find viewer script: $PY_SCRIPT" >&2
  exit 1
fi

cmd=(python3 "$PY_SCRIPT" --fps "$FPS" --start "$START")

if [[ -n "$FILE_PATH" ]]; then
  cmd+=(--file "$FILE_PATH")
else
  if [[ -z "$TASK_NAME" ]]; then
    echo "Need --task or --file" >&2
    usage
    exit 1
  fi
  cmd+=(--dataset_dir "$DATASET_DIR" --task_name "$TASK_NAME" --episode_idx "$EPISODE_IDX")
fi

[[ "$SHOW_DEPTH" -eq 1 ]] && cmd+=(--show_depth)
[[ "$SHOW_VELOCITY" -eq 1 ]] && cmd+=(--show_velocity)
[[ "$SHOW_EFFORT" -eq 1 ]] && cmd+=(--show_effort)

exec "${cmd[@]}"
