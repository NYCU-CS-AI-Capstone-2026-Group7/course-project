import os

target_path = "/usr/local/lib/python3.11/dist-packages/leisaac/enhance/datasets/lerobot_dataset_handler.py"
if not os.path.exists(target_path):
    print(f"Error: {target_path} does not exist!")
    exit(1)

with open(target_path, "r") as f:
    content = f.read()

# Replace 1: parallel_encoding (force False)
old_flush = "self._lerobot_dataset.save_episode(parallel_encoding=True)"
new_flush = "self._lerobot_dataset.save_episode(parallel_encoding=False)"
if old_flush in content:
    content = content.replace(old_flush, new_flush)
    print("Successfully replaced parallel_encoding=True with False")
else:
    print("parallel_encoding=True not found or already sequential")

# Replace 2: clear protection
old_clear = """    def clear(self):
        self._lerobot_dataset.clear_episode_buffer()"""

new_clear = """    def clear(self):
        if getattr(self._lerobot_dataset, "episode_buffer", None) is not None:
            self._lerobot_dataset.clear_episode_buffer()"""

if old_clear in content:
    content = content.replace(old_clear, new_clear)
    print("Successfully added episode_buffer check in clear()")
else:
    # Try a fallback simple replace if formatting differed
    if "self._lerobot_dataset.clear_episode_buffer()" in content:
        content = content.replace(
            "self._lerobot_dataset.clear_episode_buffer()",
            "if getattr(self._lerobot_dataset, 'episode_buffer', None) is not None:\n            self._lerobot_dataset.clear_episode_buffer()"
        )
        print("Fallback replace used for clear()")
    else:
        print("clear method modification skipped or already done")

with open(target_path, "w") as f:
    f.write(content)

print("Patching completed successfully!")
