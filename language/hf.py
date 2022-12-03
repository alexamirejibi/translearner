from distutils.dir_util import copy_tree
from huggingface_hub import Repository, snapshot_download, create_repo, get_full_repo_name


def copy_repository_template():
    # Clone the repo and extract the local path
    template_repo_id = "lewtun/distilbert-base-uncased-finetuned-squad-d5716d28"
    commit_hash = "be3eaffc28669d7932492681cd5f3e8905e358b4"
    template_repo_dir = snapshot_download(template_repo_id, revision=commit_hash)
    # Create an empty repo on the Hub
    model_name = template_repo_id.split("/")[1]
    create_repo(model_name, exist_ok=True)
    # Clone the empty repo
    new_repo_id = get_full_repo_name(model_name)
    new_repo_dir = model_name
    repo = Repository(local_dir=new_repo_dir, clone_from=new_repo_id)
    # Copy files
    copy_tree(template_repo_dir, new_repo_dir)
    # Push to Hub
    repo.push_to_hub()

copy_repository_template()