from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="AnonRes/ResEncL-OpenMind-VoCo", #
    local_dir="input/ResEncL-OpenMind-VoCo",
)