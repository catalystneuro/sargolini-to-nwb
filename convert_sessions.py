import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List
from uuid import uuid4

import numpy as np
import nwbinspector.nwbinspector
from dateutil import tz
from hdmf.backends.hdf5 import H5DataIO
from natsort import natsorted
from nwbinspector.inspector_tools import save_report, format_messages
from pynwb import NWBFile, NWBHDF5IO
from pynwb.behavior import Position
from pynwb.file import Subject
from scipy.io import loadmat


def convert_sessions_to_nwb(
        folder_path: str,
        nwbfiles_folder_path: str,
        verbose: bool = True,
):
    if not Path(nwbfiles_folder_path).exists():
        os.makedirs(nwbfiles_folder_path)

    all_sessions = collect_sessions_from_folder(folder_path=folder_path)
    for subject_id, sessions_per_subject in all_sessions.items():
        for session_id, cell_ids in sessions_per_subject.items():
            nwbfile_path = Path(nwbfiles_folder_path) / f"{subject_id}-{session_id}.nwb"
            if nwbfile_path.exists():
                continue

            # Create nwbfile
            nwbfile = start_nwb(subject_id=subject_id, session_id=session_id)

            # Add units from file
            add_units_to_nwb(nwbfile=nwbfile, unit_ids=cell_ids)

            # Add position to file
            add_position_to_nwb(nwbfile=nwbfile)

            # Write the file
            with NWBHDF5IO(nwbfile_path, "w") as io:
                io.write(nwbfile)

            if verbose:
                print(f"NWBFile created at {nwbfile_path}.")


def collect_sessions_from_folder(folder_path: str):
    unique_sessions = defaultdict(lambda: defaultdict(list))

    for file in list(Path(folder_path).glob("*.mat")):
        match = match_file_name(file_name=str(file.name))
        if match is None:
            continue
        groups = match.groupdict()
        animal_id = groups["animal_id"]
        session_id = groups["session_id"]
        cell_id = groups["cell_id"]
        if cell_id not in unique_sessions[animal_id][session_id]:
            unique_sessions[animal_id][session_id].append(cell_id)
    return unique_sessions


def match_file_name(file_name: str):
    # Define the regular expression pattern
    pattern = r"^(?P<animal_id>\d+)-(?P<session_id>\d+)[^_]*_(?P<cell_id>[tT]\d+[cC]\d+)\.mat$"
    # using the search function to find the match
    match = re.search(pattern, file_name)
    return match


def start_nwb(subject_id: str, session_id: str) -> NWBFile:
    # Add Norway timezone information to the datetime object
    tzinfo = tz.gettz("Europe/Oslo")
    # The start time of the session is arbitrary
    session_start_time = datetime(1900, 1, 1, tzinfo=tzinfo)

    nwbfile = NWBFile(
        identifier=str(uuid4()),
        session_description=(
            "This session includes spike and position times for recorded cells from a Long Evans rat that was running in a 1 x 1 meter enclosure. "
            "The cells were recorded in the dorsocaudal 25% portion of the medial entorhinal cortex (MEC)."
            "Position is given for two LEDs to enable calculation of head direction."
        ),
        experiment_description=(
            "The sample includes conjunctive cells and head direction cells from layers III and V of medial entorhinal cortex and have been published in Sargolini et al. (Science, 2006)."
        ),
        subject=Subject(
            subject_id=subject_id,
            description="A Long Evans rat.",
            species="Rattus norvegicus",  # long evans rats
            age="P3M/P5M",  # ages between 3-5 months
            sex="M",
            weight="0.35/0.45",  # 350-450 g
        ),
        session_start_time=session_start_time,
        session_id=session_id,
        related_publications="https://doi.org/10.1126/science.1125572",
        institution="Centre for the Biology of Memory, Norwegian University of Science and Technology",
        experimenter="Sargolini, Francesca",
        keywords=["medial entorhinal cortex", "spike times", "position"],
    )

    return nwbfile


def add_units_to_nwb(nwbfile: NWBFile, unit_ids: List[str]):
    # Add units
    nwbfile.add_unit_column(
        name="unit_name",
        description="The identifier of the cell, based on tetrode number and cell number.",
    )
    for unit_id in natsorted(unit_ids):
        file_path = (
            Path(folder_path)
            / f"{nwbfile.subject.subject_id}-{nwbfile.session_id}_{unit_id}.mat"
        )
        if not file_path.is_file():
            special_named_file_paths = list(
                Path(folder_path).glob(
                    f"{nwbfile.subject.subject_id}-{nwbfile.session_id}*{unit_id}.mat"  # can sometimes have +02 in the filename
                )
            )
            file_path = special_named_file_paths[0]
        mat = read_mat_file(file_path=file_path)
        unit_name = file_path.stem.split("_")[1].lower()
        spike_times_column_name = "cellTS" if "cellTS" in mat else "ts"
        nwbfile.add_unit(spike_times=mat[spike_times_column_name], unit_name=unit_name)


def add_position_to_nwb(nwbfile: NWBFile):
    file_paths = list(
        Path(folder_path).glob(
            f"{nwbfile.subject.subject_id}-{nwbfile.session_id}*.mat"
        )
    )
    if "all_data" in folder_path:
        file_paths = list(
            Path(folder_path).glob(
                f"{nwbfile.subject.subject_id}-{nwbfile.session_id}*POS.mat"
            )
        )

    mat = read_mat_file(file_path=file_paths[0])

    position = Position(name="Position")

    # add position for first tracking LED
    position_data_led1 = []

    timestamps_column_name = "post" if "post" in mat else "t"
    timestamps = mat[timestamps_column_name]

    # x1 array with the x-positions for the first tracking LED.
    x1_position_column_name = "posx" if "posx" in mat else "x1"
    position_data_led1.append(mat[x1_position_column_name])

    # y1 array with the y-positions for the first tracking LED.
    y1_position_column_name = "posy" if "posy" in mat else "y1"
    position_data_led1.append(mat[y1_position_column_name])

    position.create_spatial_series(
        name="SpatialSeriesLED1",
        description="Position (x, y) for the first tracking LED.",
        data=H5DataIO(np.array(position_data_led1).T, compression=True),
        timestamps=H5DataIO(timestamps, compression=True),
        unit="meters",
        conversion=0.01,
        reference_frame="(0,0) is not known.",
    )

    # add position for second tracking LED
    position_data_led2 = []

    # x2 Array with the x-positions for the second tracking LED.
    # Not all sessions have two tracking LEDs
    x2_position_column_name = "posx2" if "posx2" in mat else "x2"
    if len(mat[x2_position_column_name]):
        position_data_led2.append(mat[x2_position_column_name])

    # Array with the y-positions for the second tracking LED.
    y2_position_column_name = "posy2" if "posy2" in mat else "y2"
    if len(mat[y2_position_column_name]):
        position_data_led2.append(mat[y2_position_column_name])

    if position_data_led2:
        position.create_spatial_series(
            name="SpatialSeriesLED2",
            description="Position (x, y) for the second tracking LED.",
            data=H5DataIO(np.array(position_data_led2).T, compression=True),
            timestamps=H5DataIO(timestamps, compression=True),
            unit="meters",
            conversion=0.01,
            reference_frame="(0,0) is not known.",
        )

    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="Processed behavioral data."
    )
    behavior_module.add(position)


def read_mat_file(file_path):
    return loadmat(file_path, squeeze_me=True)


if __name__ == "__main__":
    # The path to the folder that contains the .mat files
    folder_path = "/Volumes/t7-ssd/8F6BE356-3277-475C-87B1-C7A977632DA7_1/all_data"
    # The path to fhe folder where the nwb files are created
    nwbfiles_folder_path = "/Volumes/t7-ssd/sargolini_to_nwb/nwbfiles/all_data"

    convert_sessions_to_nwb(
        folder_path=folder_path, nwbfiles_folder_path=nwbfiles_folder_path
    )

    # Inspect files
    results = list(nwbinspector.inspect_all(folder_path))
    report_path = Path(nwbfiles_folder_path) / "inspector_result.txt"
    save_report(
        report_file_path=report_path,
        formatted_messages=format_messages(
            results,
            levels=["importance", "file_path"],
        ),
    )
