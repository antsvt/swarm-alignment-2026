import json
import subprocess
import sys


def test_kaggle_b1_bundle_is_readable_and_complete(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "scripts/build_kaggle_notebook.py",
            "--runner",
            "b1",
            "--output",
            str(tmp_path),
            "--kernel-id",
            "owner/swarmbench-b1-test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    notebook = json.loads((tmp_path / "swarmbench_b1.ipynb").read_text(encoding="utf-8"))
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "%%writefile /kaggle/working/swarm_alignment/benchmark.py" in source
    assert "%%writefile /kaggle/working/configs/math500-subset.json" in source
    assert "%%writefile /kaggle/working/run_kaggle_b1.py" in source
    assert "%run /kaggle/working/run_kaggle_b1.py" in source
    assert "import base64" not in source
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["enable_gpu"] and metadata["enable_internet"]
