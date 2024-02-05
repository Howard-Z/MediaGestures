from os.path import abspath, join, isdir, exists, dirname


def get_root_dir():
    current_path = abspath(".")

    while True:
        git_path = join(current_path, ".git")
        if exists(git_path) and isdir(git_path):
            return current_path

        # Move up one level in the directory hierarchy
        parent_path = dirname(current_path)

        # Check if we have reached the root of the file system
        if current_path == parent_path:
            print("No .git directory found.")
            return None

        current_path = parent_path
