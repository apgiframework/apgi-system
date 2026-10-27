"""BIDS-EEG export of a finished recording.

Writes a BIDS 1.9 raw EEG dataset using the BrainVision format (``.vhdr`` /
``.vmrk`` / ``.eeg``), one of the formats the specification allows, so no
third-party writer is needed. Per recording it produces ``_eeg.json``,
``_channels.tsv``, ``_events.tsv`` and ``_events.json``; per dataset it keeps
``dataset_description.json``, ``participants.tsv`` / ``.json`` and ``README``.

Marker onsets are stamped on the host clock; they are converted to seconds from
the first stored sample through the device clock mapping, so the events file is
aligned to the data rather than to wall-clock time.

Dropped chunks leave gaps. Gap samples are written as ``gap_fill_value`` and each
gap is also an ``events.tsv`` row with ``trial_type`` ``gap``, so analysis code
can exclude them; they are never silently interpolated.

:func:`validate_bids_eeg` is a structural self-check, not a substitute for the
official ``bids-validator``.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .clock import ClockMapper
from .errors import ExportError
from .recorder import LocalRecorder, Recording

BIDS_VERSION = "1.9.0"
_LABEL = re.compile(r"^[A-Za-z0-9]+$")
_NA = "n/a"
_CHANNEL_TYPES = {"EEG", "EOG", "ECG", "EMG", "MISC", "TRIG"}


@dataclass
class BidsExportResult:
    """Paths written for one recording."""

    root: Path
    header: Path
    data: Path
    markers: Path
    sidecar: Path
    channels: Path
    events: Path
    events_json: Path
    n_events: int
    n_gaps: int
    dropped_markers: List[str] = field(default_factory=list)

    @property
    def files(self) -> List[Path]:
        return [
            self.header,
            self.data,
            self.markers,
            self.sidecar,
            self.channels,
            self.events,
            self.events_json,
        ]


def export_session_to_bids(
    session_dir: Path,
    bids_root: Path,
    *,
    subject: str,
    task: str,
    mapper: Optional[ClockMapper] = None,
    **kwargs: Any,
) -> BidsExportResult:
    """Read a recording folder and export it. Repairs an unfinished one first."""
    session_dir = Path(session_dir)
    if not (session_dir / "COMPLETE").exists():
        LocalRecorder.recover(session_dir)
    recording = LocalRecorder.read(session_dir)
    return export_recording_to_bids(
        recording, bids_root, subject=subject, task=task, mapper=mapper, **kwargs
    )


def export_recording_to_bids(
    recording: Recording,
    bids_root: Path,
    *,
    subject: str,
    task: str,
    session: Optional[str] = None,
    run: Optional[int] = None,
    mapper: Optional[ClockMapper] = None,
    power_line_hz: Optional[float] = None,
    eeg_reference: str = _NA,
    dataset_name: str = "APGI recordings",
    task_description: str = "",
    participant: Optional[Dict[str, str]] = None,
    gap_fill_value: float = 0.0,
    overwrite: bool = False,
) -> BidsExportResult:
    """Write one recording into ``bids_root``.

    Args:
        mapper: Device-to-host clock mapping. Built from the probes saved with
            the recording when omitted.
        power_line_hz: Mains frequency (50 or 60); ``n/a`` when unknown.
        eeg_reference: Reference description, ``n/a`` when unknown.
        participant: Extra ``participants.tsv`` columns for this subject.
        overwrite: Replace existing files for this subject/session/task/run.
    """
    for label_name, value in (("subject", subject), ("session", session), ("task", task)):
        if value is not None and not _LABEL.match(value):
            raise ExportError(f"{label_name} label {value!r} must be alphanumeric (BIDS)")
    if run is not None and run < 1:
        raise ExportError("run must be a positive integer")
    if mapper is None:
        mapper = _mapper_from_recording(recording)

    root = Path(bids_root)
    stem = f"sub-{subject}" + (f"_ses-{session}" if session else "")
    stem += f"_task-{task}" + (f"_run-{run}" if run is not None else "")
    eeg_dir = root / f"sub-{subject}" / (f"ses-{session}" if session else "") / "eeg"
    paths = {
        "header": eeg_dir / f"{stem}_eeg.vhdr",
        "data": eeg_dir / f"{stem}_eeg.eeg",
        "markers": eeg_dir / f"{stem}_eeg.vmrk",
        "sidecar": eeg_dir / f"{stem}_eeg.json",
        "channels": eeg_dir / f"{stem}_channels.tsv",
        "events": eeg_dir / f"{stem}_events.tsv",
        "events_json": eeg_dir / f"{stem}_events.json",
    }
    if not overwrite and any(p.exists() for p in paths.values()):
        raise ExportError(f"{stem} already exists in {root}; pass overwrite=True to replace it")
    eeg_dir.mkdir(parents=True, exist_ok=True)

    info = recording.info
    fs = info.sampling_rate_hz
    samples = recording.samples.copy()
    for start, length in recording.gaps:
        samples[start : start + length] = gap_fill_value

    events, dropped = _build_events(recording, mapper)
    n_gaps = len(recording.gaps)

    _write_data(paths["data"], samples)
    _write_vhdr(paths["header"], paths["data"].name, paths["markers"].name, recording)
    _write_vmrk(paths["markers"], paths["data"].name, events, fs)
    _write_json(
        paths["sidecar"],
        _eeg_sidecar(recording, task, task_description, power_line_hz, eeg_reference),
    )
    _write_tsv(
        paths["channels"],
        ["name", "type", "units", "sampling_frequency", "status"],
        [
            [
                c.name,
                c.kind if c.kind in _CHANNEL_TYPES else "MISC",
                c.unit if c.unit != "uV" else "µV",
                f"{fs:g}",
                "good",
            ]
            for c in info.channels
        ],
    )
    _write_tsv(
        paths["events"],
        ["onset", "duration", "trial_type", "value", "sample", "source"],
        [
            [
                f"{e['onset']:.6f}",
                f"{e['duration']:.6f}",
                e["trial_type"],
                e["value"],
                str(e["sample"]),
                e["source"],
            ]
            for e in events
        ],
    )
    _write_json(paths["events_json"], _events_sidecar())
    _ensure_dataset_files(root, dataset_name)
    _update_participants(root, subject, participant or {})

    return BidsExportResult(
        root=root,
        header=paths["header"],
        data=paths["data"],
        markers=paths["markers"],
        sidecar=paths["sidecar"],
        channels=paths["channels"],
        events=paths["events"],
        events_json=paths["events_json"],
        n_events=len(events),
        n_gaps=n_gaps,
        dropped_markers=dropped,
    )


# -- events ------------------------------------------------------------------
def _mapper_from_recording(recording: Recording) -> ClockMapper:
    if not recording.sync_probes:
        raise ExportError(
            "the recording has no clock-sync probes; pass an explicit mapper so marker "
            "times can be aligned to the data"
        )
    mapper = ClockMapper()
    for probe in recording.sync_probes:
        mapper.add_probe(probe)
    return mapper


def _build_events(
    recording: Recording, mapper: ClockMapper
) -> Tuple[List[Dict[str, Any]], List[str]]:
    fs = recording.info.sampling_rate_hz
    t0 = recording.sample0_device_time_s
    duration_s = recording.n_samples / fs
    events: List[Dict[str, Any]] = []
    dropped: List[str] = []
    for marker in recording.markers:
        onset = mapper.host_to_device_s(marker.onset_host_ns) - t0
        if onset < 0 or onset > duration_s:
            dropped.append(marker.marker_id)
            continue
        if marker.code is not None:
            value = str(marker.code)
        elif marker.value is not None:
            value = marker.value
        else:
            value = _NA
        events.append(
            {
                "onset": onset,
                "duration": marker.duration_s,
                "trial_type": marker.label,
                "value": value,
                "sample": int(round(onset * fs)),
                "source": marker.source,
            }
        )
    for start, length in recording.gaps:
        events.append(
            {
                "onset": start / fs,
                "duration": length / fs,
                "trial_type": "gap",
                "value": _NA,
                "sample": start,
                "source": "recorder",
            }
        )
    events.sort(key=lambda e: (e["onset"], e["trial_type"]))
    return events, dropped


# -- BrainVision -------------------------------------------------------------
def _bv_text(text: str) -> str:
    """BrainVision escapes commas as ``\\1`` and has no multi-line fields."""
    return " ".join(text.split()).replace(",", "\\1")


def _write_data(path: Path, samples: Any) -> None:
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as fh:
        fh.write(np.ascontiguousarray(samples, dtype="<f4").tobytes())
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _write_vhdr(path: Path, data_name: str, marker_name: str, recording: Recording) -> None:
    info = recording.info
    lines = [
        "Brain Vision Data Exchange Header File Version 1.0",
        "; Written by the APGI acquisition exporter",
        "",
        "[Common Infos]",
        "Codepage=UTF-8",
        f"DataFile={data_name}",
        f"MarkerFile={marker_name}",
        "DataFormat=BINARY",
        "; Data orientation: MULTIPLEXED=ch1,pt1, ch2,pt1 ...",
        "DataOrientation=MULTIPLEXED",
        f"NumberOfChannels={info.n_channels}",
        "; Sampling interval in microseconds",
        f"SamplingInterval={1e6 / info.sampling_rate_hz:.10g}",
        "",
        "[Binary Infos]",
        "BinaryFormat=IEEE_FLOAT_32",
        "",
        "[Channel Infos]",
        '; Each entry: Ch<Channel number>=<Name>,<Reference channel name>,<Resolution in "Unit">,<Unit>',
    ]
    for i, ch in enumerate(info.channels, start=1):
        unit = "µV" if ch.unit == "uV" else ch.unit
        lines.append(f"Ch{i}={_bv_text(ch.name)},,1,{_bv_text(unit)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_vmrk(
    path: Path, data_name: str, events: List[Dict[str, Any]], sampling_rate_hz: float
) -> None:
    lines = [
        "Brain Vision Data Exchange Marker File, Version 1.0",
        "",
        "[Common Infos]",
        "Codepage=UTF-8",
        f"DataFile={data_name}",
        "",
        "[Marker Infos]",
        "; Each entry: Mk<Marker number>=<Type>,<Description>,<Position in data points>,",
        "; <Size in data points>, <Channel number (0 = marker is related to all channels)>",
        "Mk1=New Segment,,1,1,0",
    ]
    for n, event in enumerate(events, start=2):
        kind = "Comment" if event["trial_type"] == "gap" else "Stimulus"
        size = max(1, int(round(event["duration"] * sampling_rate_hz)))  # in data points
        lines.append(f"Mk{n}={kind},{_bv_text(event['trial_type'])},{event['sample'] + 1},{size},0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -- sidecars ----------------------------------------------------------------
def _eeg_sidecar(
    recording: Recording,
    task: str,
    description: str,
    power_line_hz: Optional[float],
    reference: str,
) -> Dict[str, Any]:
    info = recording.info
    counts = {k: 0 for k in _CHANNEL_TYPES}
    for ch in info.channels:
        counts[ch.kind if ch.kind in _CHANNEL_TYPES else "MISC"] += 1
    sidecar: Dict[str, Any] = {
        "TaskName": task,
        "SamplingFrequency": info.sampling_rate_hz,
        "EEGReference": reference,
        "SoftwareFilters": _NA,
        "PowerLineFrequency": power_line_hz if power_line_hz is not None else _NA,
        "EEGChannelCount": counts["EEG"],
        "EOGChannelCount": counts["EOG"],
        "ECGChannelCount": counts["ECG"],
        "EMGChannelCount": counts["EMG"],
        "MiscChannelCount": counts["MISC"],
        "TriggerChannelCount": counts["TRIG"],
        "RecordingDuration": recording.n_samples / info.sampling_rate_hz,
        "RecordingType": "continuous",
        "ManufacturersModelName": info.model,
        "SoftwareVersions": f"apgi-framework acquisition (device driver {info.driver})",
    }
    if description:
        sidecar["TaskDescription"] = description
    if info.firmware:
        sidecar["DeviceFirmware"] = info.firmware
    if info.serial:
        sidecar["DeviceSerialNumber"] = info.serial
    return sidecar


def _events_sidecar() -> Dict[str, Any]:
    return {
        "onset": {
            "Description": "Seconds from the first stored sample, via the device clock mapping",
            "Units": "s",
        },
        "duration": {"Description": "Event duration", "Units": "s"},
        "trial_type": {"Description": "Marker label; 'gap' marks lost samples"},
        "value": {"Description": "Trigger code or marker value"},
        "sample": {"Description": "Zero-based sample index of the onset"},
        "source": {"Description": "software, hardware, or recorder"},
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_tsv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def _ensure_dataset_files(root: Path, name: str) -> None:
    from apgi_framework import __version__

    description = root / "dataset_description.json"
    if not description.exists():
        _write_json(
            description,
            {
                "Name": name,
                "BIDSVersion": BIDS_VERSION,
                "DatasetType": "raw",
                "GeneratedBy": [{"Name": "apgi-framework", "Version": __version__}],
            },
        )
    readme = root / "README"
    if not readme.exists():
        readme.write_text(
            f"# {name}\n\nEEG recordings exported by the APGI acquisition subsystem.\n"
            "Events are aligned to the data through a device-to-host clock mapping.\n",
            encoding="utf-8",
        )
    participants_json = root / "participants.json"
    if not participants_json.exists():
        _write_json(participants_json, {"participant_id": {"Description": "Subject label"}})


def _update_participants(root: Path, subject: str, extra: Dict[str, str]) -> None:
    path = root / "participants.tsv"
    rows: Dict[str, Dict[str, str]] = {}
    columns = ["participant_id"]
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            columns = list(reader.fieldnames or columns)
            for row in reader:
                rows[row["participant_id"]] = dict(row)
    entry = rows.setdefault(f"sub-{subject}", {"participant_id": f"sub-{subject}"})
    for key, value in extra.items():
        if key not in columns:
            columns.append(key)
        entry[key] = value
    _write_tsv(
        path,
        columns,
        [[row.get(c, _NA) for c in columns] for _, row in sorted(rows.items())],
    )


# -- validation --------------------------------------------------------------
def validate_bids_eeg(root: Path) -> List[str]:
    """Structural checks on an exported dataset; returns problems (empty means none found)."""
    root = Path(root)
    problems: List[str] = []
    description = root / "dataset_description.json"
    if not description.exists():
        problems.append("missing dataset_description.json")
    else:
        data = json.loads(description.read_text(encoding="utf-8"))
        for key in ("Name", "BIDSVersion"):
            if key not in data:
                problems.append(f"dataset_description.json lacks {key}")
    if not (root / "participants.tsv").exists():
        problems.append("missing participants.tsv")
    headers = sorted(root.glob("sub-*/**/eeg/*_eeg.vhdr"))
    if not headers:
        problems.append("no *_eeg.vhdr files found")
    for header in headers:
        problems.extend(_validate_recording(root, header))
    return problems


def _validate_recording(root: Path, header: Path) -> List[str]:
    problems: List[str] = []
    stem = header.name[: -len("_eeg.vhdr")]
    rel = header.relative_to(root)
    text = header.read_text(encoding="utf-8")
    fields = dict(
        line.split("=", 1) for line in text.splitlines() if "=" in line and not line.startswith(";")
    )
    n_channels = int(fields.get("NumberOfChannels", "0"))
    for key in ("DataFile", "MarkerFile"):
        name = fields.get(key, "")
        if not name or not (header.parent / name).exists():
            problems.append(f"{rel}: {key} {name!r} not found next to the header")
    data_name = fields.get("DataFile", "")
    if data_name and not data_name.startswith(stem):
        problems.append(f"{rel}: DataFile {data_name!r} does not follow the BIDS name {stem}")
    data_path = header.parent / data_name
    if data_path.exists() and n_channels and data_path.stat().st_size % (4 * n_channels):
        problems.append(f"{rel}: data size is not a multiple of channels x 4 bytes")
    sidecar = header.parent / f"{stem}_eeg.json"
    if not sidecar.exists():
        problems.append(f"{rel}: missing {sidecar.name}")
    else:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        for key in (
            "TaskName",
            "SamplingFrequency",
            "EEGReference",
            "SoftwareFilters",
            "PowerLineFrequency",
        ):
            if key not in meta:
                problems.append(f"{sidecar.name} lacks required field {key}")
    channels = header.parent / f"{stem}_channels.tsv"
    if not channels.exists():
        problems.append(f"{rel}: missing {channels.name}")
    else:
        with open(channels, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        if len(rows) != n_channels:
            problems.append(f"{channels.name}: {len(rows)} rows but header declares {n_channels}")
        for row in rows:
            if row.get("type") not in _CHANNEL_TYPES:
                problems.append(f"{channels.name}: invalid type {row.get('type')!r}")
    events = header.parent / f"{stem}_events.tsv"
    if not events.exists():
        problems.append(f"{rel}: missing {events.name}")
    else:
        with open(events, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            if (reader.fieldnames or [])[:2] != ["onset", "duration"]:
                problems.append(f"{events.name}: first columns must be onset, duration")
            for row in reader:
                if float(row["onset"]) < 0:
                    problems.append(f"{events.name}: negative onset {row['onset']}")
    return problems
