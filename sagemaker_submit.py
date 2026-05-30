"""Submit a SageMaker training job for this project.
Defaults pre-filled from Stage 1 scan. Review before running real jobs (spot saves 60-70%).
Usage: python sagemaker_submit.py [--samples N] [--instance ml.g6.xlarge] [--spot] [--no-wait]
"""
import argparse, os, shutil, tempfile, time
import boto3
import sagemaker
from sagemaker.estimator import Estimator


# Whitelist: only these paths get uploaded as source. Keeps source_dir small
# even when the project has gigabytes of outputs/ or model caches.
# Adapters/probes are included as directories so their *configs* travel with the job;
# the ignore patterns below filter out weight files (you'll want big adapter weights
# pre-uploaded to S3 and referenced via --adapter s3://... instead).
SOURCE_WHITELIST = (
    "shared", "blue_team", "red_team", "benchmarks", "scripts",
    "adapters", "probes", "calibrations",
    "requirements-shared.txt", "requirements-cuda.txt", "requirements-mlx.txt",
    "pyproject.toml", "setup.py", "README.md",
)


def stage_source_dir(repo_root: str) -> str:
    """Copy only whitelisted paths to a temp dir. Avoids the 'cwd is 2GB, upload takes forever' trap."""
    staging = tempfile.mkdtemp(prefix="sm-source-")
    for name in SOURCE_WHITELIST:
        src = os.path.join(repo_root, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(staging, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".pytest_cache",
                "*.npz", "*.pt", "*.bin", "*.safetensors"))
        else:
            shutil.copy2(src, dst)
    return staging


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instance", default=os.environ.get("INSTANCE_TYPE", "ml.g6.xlarge"))
    p.add_argument("--max-runtime-hours", type=int, default=4)
    p.add_argument("--spot", action="store_true", help="use spot instances (60-70%% cheaper; safe if your script is resume-friendly)")
    p.add_argument("--no-wait", action="store_true", help="submit async; poll status with describe-training-job")
    p.add_argument("--job-name", default=None)
    p.add_argument("--entry-point", default="scripts/run_pillar.py",
                   help="path (relative to repo root) of the script to run on the training instance")
    args = p.parse_args()

    region = os.environ["AWS_REGION"]
    bucket = os.environ["S3_BUCKET"]
    handle = os.environ["STUDENT_HANDLE"]
    project = os.environ["PROJECT_SLUG"]
    account = boto3.client("sts").get_caller_identity()["Account"]
    role_arn = f"arn:aws:iam::{account}:role/AmazonSageMaker-{handle}"

    # AWS Deep Learning Container — PyTorch 2.4 / py311 / CUDA 12.4.
    # If your project pins a newer torch (e.g. 2.8), the container will pip-install it over
    # the bundled 2.4 at cold start (~2-3 min added). Alternatives: relax your torch pin to
    # '>=2.4', or build a custom ECR image. Image catalog: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
    image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.4.0-gpu-py311-cu124-ubuntu22.04-sagemaker"

    job_name = args.job_name or f"{project}-{int(time.time())}"
    staging_dir = stage_source_dir(os.getcwd())
    print(f"Submitting {job_name}")
    print(f"  source:   {staging_dir} (staged from {os.getcwd()}, whitelist only)")
    print(f"  entry:    {args.entry_point}")
    print(f"  instance: {args.instance}{' (spot)' if args.spot else ' (on-demand)'}")
    print(f"  max run:  {args.max_runtime_hours}h")

    estimator = Estimator(
        image_uri=image_uri,
        role=role_arn,
        instance_type=args.instance,
        instance_count=1,
        volume_size=100,
        max_run=args.max_runtime_hours * 3600,
        use_spot_instances=args.spot,
        max_wait=(args.max_runtime_hours * 3600 + 3600) if args.spot else None,
        checkpoint_s3_uri=f"s3://{bucket}/checkpoints/{job_name}/" if args.spot else None,
        sagemaker_session=sagemaker.Session(boto_session=boto3.Session(region_name=region)),
        output_path=f"s3://{bucket}/runs/{job_name}/",
        base_job_name=project,
        entry_point=args.entry_point,
        source_dir=staging_dir,
        environment={
            "HF_TOKEN": os.environ["HF_TOKEN"],
            "S3_BUCKET": bucket,
            "HF_HOME": "/opt/ml/input/data/hf_cache",
            "TRANSFORMERS_CACHE": "/opt/ml/input/data/hf_cache",
        },
        tags=[
            {"Key": "participant", "Value": handle},
            {"Key": "project", "Value": project},
            {"Key": "compute_path", "Value": "A"},
        ],
    )
    estimator.fit(job_name=job_name, wait=not args.no_wait, logs="All" if not args.no_wait else None)
    if args.no_wait:
        print(f"Submitted async. Poll: aws sagemaker describe-training-job --training-job-name {job_name} --region {region}")
    else:
        print(f"Done. Pull: aws s3 sync s3://{bucket}/runs/{job_name}/ ./outputs/{job_name}/")


if __name__ == "__main__":
    main()
