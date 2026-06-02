"""Utility script to merge multiple UMI-style object_poses.json files.

Combines lists of episode poses and automatically resolves potential video_name
conflicts by appending suffixes or renaming them to avoid downstream conflicts
in the simulator or recorder.

Usage:
    python scripts/datagen/merge_poses.py \
        --inputs data/procedural_spawn/demos/mapping/object_poses.json data/AI-final-49/demos/mapping/object_poses.json \
        --output data/merged_object_poses.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge multiple object_poses.json files.")
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="List of paths to input object_poses.json files to merge.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output merged object_poses.json file.",
    )
    parser.add_argument(
        "--resolve_conflicts",
        action="store_true",
        default=True,
        help="Automatically resolve video_name conflicts by renaming duplicates.",
    )
    args = parser.parse_args()

    merged_episodes = []
    seen_video_names = set()
    conflict_count = 0

    print("=== Start Merging Object Poses ===")
    for input_str in args.inputs:
        input_path = Path(input_str)
        if not input_path.exists():
            print(f"[Error] File not found: {input_path}")
            continue

        try:
            with open(input_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Error] Failed to parse JSON from {input_path}: {e}")
            continue

        if not isinstance(data, list):
            print(f"[Warning] Expected a list of episodes in {input_path}, got {type(data)}. Skipping.")
            continue

        file_added_count = 0
        for ep in data:
            if not isinstance(ep, dict) or "video_name" not in ep:
                # Skip invalid entries
                continue

            new_ep = ep.copy()
            video_name = new_ep["video_name"]

            # If conflict, resolve it by appending suffix
            if video_name in seen_video_names:
                if args.resolve_conflicts:
                    suffix_counter = 1
                    new_video_name = f"{video_name}_dup{suffix_counter}"
                    while new_video_name in seen_video_names:
                        suffix_counter += 1
                        new_video_name = f"{video_name}_dup{suffix_counter}"
                    
                    print(f"  [Conflict] Rename duplicate video_name: '{video_name}' -> '{new_video_name}' in {input_path.name}")
                    new_ep["video_name"] = new_video_name
                    video_name = new_video_name
                    conflict_count += 1
                else:
                    print(f"  [Warning] Duplicate video_name found: '{video_name}' in {input_path.name} (ignoring because --no-resolve-conflicts)")

            seen_video_names.add(video_name)
            merged_episodes.append(new_ep)
            file_added_count += 1

        print(f"Loaded {file_added_count} episodes from {input_path.name}")

    # Save to output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, "w") as f:
            json.dump(merged_episodes, f, indent=4)
        print("---------------------------------")
        print(f"Successfully merged {len(args.inputs)} files into: {output_path.absolute()}")
        print(f"Total merged episodes: {len(merged_episodes)}")
        print(f"Total naming conflicts resolved: {conflict_count}")
    except Exception as e:
        print(f"[Error] Failed to write merged JSON to {output_path}: {e}")


if __name__ == "__main__":
    main()
