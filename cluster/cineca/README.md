# CINECA GPU cluster setup

This directory contains the isolated CINECA access and GPU smoke-test setup for the remaining LuLu supplementary work. It deliberately does **not** contain a password, OTP, recovery code, SSH private key, or temporary certificate.

## Identity mapping

- Smallstep/OIDC certificate identity: `boliang.liu@campus.tu-berlin.de`
- CINECA SSH username / certificate Principal: `bliu0001`
- Target CINECA cluster: Leonardo (`login.leonardo.cineca.it`)
- Active Project Account: `AIFAC_S07_051` (2026-09-15 through 2027-03-15)
- GPU partition / debug QoS: `boost_usr_prod` / `boost_qos_dbg`

## Verified access and queue probe

On 2026-09-19, certificate-based SSH access was verified without a second OTP prompt. A real one-node, four-GPU debug smoke job was submitted as Slurm job `58223404`:

- submit: `2026-09-19T17:55:31` (cluster local time)
- start: `2026-09-19T17:55:51`
- measured Slurm queue wait: 20 seconds
- allocated node: `lrdn0145`
- allocated TRES: `gres/gpu=4`, `cpu=8`, `mem=123200M`, `node=1`
- log: `$HOME/lulu_cluster_smoke/lulu-4gpu-probe-58223404.out`

This is a point-in-time observation, not a guaranteed future queue time.

## Verified persistence across Slurm jobs

Persistence was tested with two separate completed four-GPU allocations. Job `58223404` ended before job `58223751` started. The second job:

- read the first job's log from `$HOME/lulu_cluster_smoke`;
- read a marker created on login node `login05.leonardo.local` under `$WORK`;
- executed the Python environment stored in the same `$WORK` directory;
- saw four NVIDIA A100-SXM-64GB devices on compute node `lrdn0426.leonardo.local`;
- wrote new files into `$WORK`, which were read again from the login node after the job completed.

Persistent test directory:

```text
/leonardo_work/AIFAC_S07_051/lulu_persistence_probe_20260920
```

Therefore Slurm allocations, processes, and GPU access are temporary, while files under `$HOME` and `$WORK` remain available to later allocations. `$TMPDIR` is job-local and must not hold durable outputs. No default `conda` command or Conda module was visible during the probe. A Conda installation and its environments will persist on the same principle if their installation root and `envs_dirs` are placed under `$WORK` or `$HOME`; they will not persist if created under `$TMPDIR`.

The email and HPC username serve different purposes. CINECA's documented sequence is:

```bash
step ssh login 'boliang.liu@campus.tu-berlin.de' --provisioner cineca-hpc
step ssh list --raw 'boliang.liu@campus.tu-berlin.de' | step ssh inspect
ssh bliu0001@login.leonardo.cineca.it
```

Do not infer the SSH username from the email address or from unverified screenshots.

## Scope

- The already completed baseline experiments are read-only inputs and must not be resubmitted.
- This setup first proves Smallstep authentication, SSH, Slurm account discovery, and GPU job validation.
- It does not submit a GPU job unless `--submit` is explicitly supplied.
- Production LuLu runs are not yet enabled. The current LuLu controller assumes eight GPUs on one host, while Leonardo Booster nodes expose four GPUs per node; that topology must be reconciled before expensive jobs are launched.

## One-time local setup

```bash
cd cluster/cineca
cp cluster.env.example cluster.env
scripts/install_step.sh
```

The local `cluster.env` records the confirmed CINECA UserDB email. Obtain the 12-hour certificate with:

```bash
scripts/bootstrap_smallstep.sh
```

The browser authentication step asks for the CINECA password and OTP. These values are sent only to CINECA and are not saved here.

## Verify access and discover the Slurm project

```bash
scripts/connect.sh
scripts/probe_cluster.sh
```

Copy the discovered project account into `CINECA_ACCOUNT` in `cluster.env`. If the account is assigned to another CINECA machine, update `CINECA_HOST`, `ssh_config`, partition, and QoS from the probe results before continuing.

## Validate the GPU request without allocating a GPU

```bash
scripts/submit_gpu_smoke.sh --test-only
```

Only after reviewing the resolved account/partition/QoS should the smoke job be submitted:

```bash
scripts/submit_gpu_smoke.sh --submit
```

## Certificate renewal

CINECA certificates expire after 12 hours (and are lost if the local SSH agent is stopped). Renew with:

```bash
scripts/bootstrap_smallstep.sh
```

The local tool, CA state, SSH-agent metadata, user-specific `cluster.env`, and logs live under ignored paths.

## Eight-GPU verl container setup

Leonardo Booster nodes expose four A100 GPUs each. An eight-GPU job therefore
uses two nodes. The setup job requests `2 x 4` GPUs, pulls the official stable
`verlai/verl:uv.cu130` Docker image as an Apptainer/Singularity SIF, stores it
under `$WORK/lulu_verl/images`, and validates PyTorch CUDA visibility on both
nodes:

```bash
scripts/submit_verl_8gpu_setup.sh
```

The persistent layout is:

```text
$WORK/lulu_verl/
  images/       # immutable SIF images
  cache/        # Apptainer and Hugging Face caches
  data/         # training data supplied later
  models/       # input model snapshots supplied later
  checkpoints/  # durable training outputs
  logs/         # Slurm logs
  src/          # training source/configuration
```

This setup job validates the container only. It does not start the LuLu/verl
training run until the dataset, model paths, and training parameters are fixed.
