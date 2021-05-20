from typing import *

import os
import sys
from pathlib import Path
import itertools

import subprocess

import fire


class Constants:
    global_cluster_path = ""
    global_cluster_command = "{python} {script_path} {data_dir} {out_dir}"

    run_script_path = "{system}.sh"
    log_path = "{system}.log"
    output_path = "{system}.out"
    error_path = "{system}.err"

    submit_script = (
        "#################### \n"
        "# \n"
        "# Example 1                                   \n"
        "# Simple HTCondor submit description file \n"
        "#                          \n"
        "####################    \n"

        "Executable   = {executable_file} \n"
        "Log          = {log_file} \n"
        "Output = {output_file} \n"
        "Error = {error_file} \n"

        "request_memory = {request_memory} GB \n"

        "Queue"
    )
    submit_script_name = "{system}.job"

    chmod_command = "chmod 777 {path}"

    submit_command = "condor_submit {submit_script_path}"


def execute(command: str):
    p = subprocess.Popen(
        command,
        shell=True,
    )
    p.communicate()


def get_command(data_dir: Path, out_dir: Path):
    script_path = Path(os.path.realpath(__file__)).parent.parent / "pandda_global_cluster.py"

    python = sys.executable

    command = Constants.global_cluster_command.format(
        python=python,
        script_path=str(script_path),
        data_dir=str(data_dir),
        out_dir=str(out_dir),
    )

    return command


def write(string: str, path: Path):
    with open(str(path), "w") as f:
        f.write(string)


def chmod(path: Path):
    subprocess.Popen(
        Constants.chmod_command.format(path=str(path)),
        shell=True,
    )


def get_run_script_path(dataset_dir: Path, out_dir: Path):
    run_script_path = out_dir / Constants.run_script_path.format(system=dataset_dir.name)
    return run_script_path


def get_log_path(dataset_dir: Path, out_dir: Path):
    log_path = out_dir / Constants.log_path.format(system=dataset_dir.name)
    return log_path


def get_output_path(dataset_dir: Path, out_dir: Path):
    output_path = out_dir / Constants.output_path.format(system=dataset_dir.name)
    return output_path


def get_error_path(dataset_dir: Path, out_dir: Path):
    error_path = out_dir / Constants.error_path.format(system=dataset_dir.name)
    return error_path


def get_submit_script(run_script_file: Path, log_file: Path, output_file: Path, error_file: Path, request_memory: int):
    submit_script = Constants.submit_script.format(
        executable_file=run_script_file,
        log_file=log_file,
        output_file=output_file,
        error_file=error_file,
        request_memory=request_memory,
    )

    return submit_script


def get_submit_script_path(dataset_dir_path: Path, out_dir: Path):
    submit_script_path = out_dir / Constants.submit_script_name.format(system=dataset_dir_path.name)

    return submit_script_path


def get_submit_command(submit_script_path: Path):
    submit_command: str = Constants.submit_command.format(submit_script_path=submit_script_path)

    return submit_command


def run(datasets_dir: str, out_dir: str, request_memory: int, debug=True):
    #
    datasets_dir = Path(datasets_dir)
    out_dir = Path(out_dir)
    request_memory = int(request_memory)
    if debug:
        print(f"Datasets dir: {datasets_dir}; out_dir: {out_dir}; requested memory: {request_memory}")

    # Get dataset dirs
    dataset_dirs = list(datasets_dir.glob("*"))
    if debug:
        print(f"Got {len(dataset_dirs)} dataset dirs. Dataset dir example: {dataset_dirs[0]}")

    # get commands
    commands: List[str] = list(
        map(
            lambda _: get_command(*_),
            zip(dataset_dirs, [out_dir] * len(dataset_dirs)),
        ))
    if debug:
        print(f"Got {len(commands)} commands. Command example: {commands[0]}")

    # run script path
    run_script_paths: List[Path] = list(
        map(
            lambda _: get_run_script_path(*_),
            zip(dataset_dirs, [out_dir] * len(dataset_dirs)),
        )
    )
    if debug:
        print(f"Got {len(run_script_paths)} run scrupt paths. run script path example: {run_script_paths[0]}")

    # write run script
    map(lambda _: write(*_), zip(commands, run_script_paths))

    # chmod
    map(chmod, run_script_paths)

    # Get log, output and error paths
    log_paths: List[Path] = list(
        map(
            lambda _: get_log_path(*_),
            zip(dataset_dirs, [out_dir]*len(dataset_dirs))
        )
    )
    if debug: print(f"Got {len(log_paths)} log paths. Log path example: {log_paths[0]}")

    output_paths: List[Path] = list(
        map(
            lambda _: get_output_path(*_),
            zip(dataset_dirs, [out_dir]*len(dataset_dirs))
        )
    )
    if debug:
        print(f"Got {len(output_paths)} output paths. Output path example: {output_paths[0]}")

    error_paths: List[Path] = list(
        map(
            lambda _: get_error_path(*_),
            zip(dataset_dirs, [out_dir]*len(dataset_dirs))
        )
    )
    if debug:
        print(f"Got {len(error_paths)} error paths. Error path example: {error_paths[0]}")

    # sumbit script
    submit_scripts: List[str] = list(
        map(
            lambda _: get_submit_script(*_),
            zip(
                run_script_paths,
                log_paths,
                output_paths,
                error_paths,
                [request_memory] * len(run_script_paths),
            ),
        ),
    )
    if debug:
        print(f"Got {len(submit_scripts)} submit scripts. submit script example: {submit_scripts[0]}")

    # tmp submit script path
    submit_script_paths: List[Path] = list(
        map(
            lambda _: get_submit_script_path(*_),
            zip(dataset_dirs, [out_dir] * len(dataset_dirs)),
        )
    )
    if debug:
        print(f"Got {len(submit_script_paths)} submit script paths. Submit script example: {submit_script_paths[0]}")

    # write submit script
    map(lambda _: write(*_), zip(submit_scripts, submit_script_paths))

    # chmod
    map(chmod, submit_script_paths)

    # Submit command
    submit_commands: List[str] = list(map(get_submit_command, submit_script_paths))
    if debug:
        print(f"Got {len(submit_commands)} submit commands. Submit command example: {submit_commands[0]}")

    # submit
    map(execute, submit_commands)

if __name__ == "__main__":
    fire.Fire(run)
