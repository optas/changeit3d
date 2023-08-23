## Bash Scripts (.sh)

These bash scripts are a quick way to re-run basically almost all of our experiments.

This code is _not_ meant to be ultra-clea and requires some "rewiring" to work in your environment. 

Importantly: these bash scripts (.sh) are **wrappers** of well the documented Python code (scripts) under /changeit3d/scripts/*.py

If you decide to run them, you need to _update the addresses of the input/output_ of each .sh file to correspond to the locations where the changeit3d repository and the downloaded ShapeTalk data reside in your machine. 

### Examples
    For instance, let's consider training a PC-AE. The script would be under Step.1 (`1_train_test_pc_ae/ablation_1.sh`). 

    If you open that script, you will see that it calls the Python script `changeit3d/scripts/train_test_pc_ae.py`; you will still have to change the
    input locations of the `code_top_dir` to point to changeIt3D and `data_top_dir` to the top dir where you stored ShapeTalk. 

    Similar things have to be done to run latent-shape-based neural listeners (Step.2), evaluating/oracle (raw) neural listeners (Step.3), a changeIt3D network (Step.4) etc.


