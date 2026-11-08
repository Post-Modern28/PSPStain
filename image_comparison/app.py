from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Dict, Set

import streamlit as st
from PIL import Image, ImageOps


st.set_page_config(
    page_title="Image Generation Comparison",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
BENCHMARKING_DIR = PROJECT_DIR.parent

# PSPStain results
PSP_RESULTS_DIR = Path(
    os.environ.get("RESULTS_DIR", PROJECT_DIR / "results")
).expanduser().resolve()

# NCE results
NCE_RESULTS_DIR = Path(
    os.environ.get(
        "NCE_RESULTS_DIR",
        BENCHMARKING_DIR / "NCE" / "results",
    )
).expanduser().resolve()

# BCI / PyramidPix2pix results
BCI_RESULTS_DIR = Path(
    os.environ.get(
        "BCI_RESULTS_DIR",
        BENCHMARKING_DIR / "BCI" / "PyramidPix2pix" / "results",
    )
).expanduser().resolve()

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"
}


# ---------------------------------------------------------
# Model configurations
# ---------------------------------------------------------
#
# For PSPStain and NCE, every source is a directory containing
# images with matching filenames.
#
# For BCI, all image types are stored in one directory and are
# distinguished by suffixes in filenames:
#   <sample>_real_A.png
#   <sample>_real_B.png
#   <sample>_fake_B.png
#
# The app removes these suffixes and matches images by <sample>.
# ---------------------------------------------------------

MODEL_CONFIGS = {
    "PSPStain": {
        "reference": "MIST",
        "results_dir": PSP_RESULTS_DIR,
        "source_specs": {
            "Input": {
                "folder": PSP_RESULTS_DIR / "MIST/val_pretrained/images/real_A",
            },
            "Ground Truth": {
                "folder": PSP_RESULTS_DIR / "MIST/val_pretrained/images/real_B",
            },
            "MIST": {
                "folder": PSP_RESULTS_DIR / "MIST/val_pretrained/images/fake_B",
            },
            "Hibou 15": {
                "folder": PSP_RESULTS_DIR / "hibou_15_patches/val_latest/images/fake_B",
            },
            "Hibou 25": {
                "folder": PSP_RESULTS_DIR / "hibou_25_patches/val_latest/images/fake_B",
            },
            "Random 15": {
                "folder": PSP_RESULTS_DIR / "random_15/val_latest/images/fake_B",
            },
        },
        "filter_mode": "filename",
    },

    "NCE": {
        "reference": "NCE Baseline",
        "results_dir": NCE_RESULTS_DIR,
        "source_specs": {
            "Input": {
                "folder": NCE_RESULTS_DIR / "NCE_baseline/val_latest/images/real_A",
            },
            "Ground Truth": {
                "folder": NCE_RESULTS_DIR / "NCE_baseline/val_latest/images/real_B",
            },
            "NCE Baseline": {
                "folder": NCE_RESULTS_DIR / "NCE_baseline/val_latest/images/fake_B",
            },
            "NCE Hibou 15": {
                "folder": NCE_RESULTS_DIR / "NCE_hibou_15/val_latest/images/fake_B",
            },
            "NCE Hibou 25": {
                "folder": NCE_RESULTS_DIR / "NCE_hibou_25/val_latest/images/fake_B",
            },
            "NCE Random 15": {
                "folder": NCE_RESULTS_DIR / "NCE_random_15/val_latest/images/fake_B",
            },
            "NCE Random 25": {
                "folder": NCE_RESULTS_DIR / "NCE_random_25/val_latest/images/fake_B",
            },
        },
        "filter_mode": "filename",
    },

    "BCI (PyramidPix2pix)": {
        "reference": "Input",
        "results_dir": BCI_RESULTS_DIR,
        "source_specs": {
            "Input": {
                "folder": BCI_RESULTS_DIR / "BCI_baseline/test_latest/images",
                "suffix": "_real_A",
            },
            "Ground Truth": {
                "folder": BCI_RESULTS_DIR / "BCI_baseline/test_latest/images",
                "suffix": "_real_B",
            },
            "BCI Baseline": {
                "folder": BCI_RESULTS_DIR / "BCI_baseline/test_latest/images",
                "suffix": "_fake_B",
            },
            "BCI Hibou 15": {
                "folder": BCI_RESULTS_DIR / "BCI_hibou_15/test_latest/images",
                "suffix": "_fake_B",
            },
            "BCI Hibou 25": {
                "folder": BCI_RESULTS_DIR / "BCI_hibou_25/test_latest/images",
                "suffix": "_fake_B",
            },
            "BCI Random 15": {
                "folder": BCI_RESULTS_DIR / "BCI_random_15/test_latest/images",
                "suffix": "_fake_B",
            },
        },
        "filter_mode": "stem",
    },
}


# Subsets are shared between PSPStain, NCE, and BCI.
# They are defined by filenames from the PSPStain MIST evaluation set.
FILTER_DIRS = {
    "Comb": PSP_RESULTS_DIR / "MIST/val_pretrained/comb/fake_B",
    "Good": PSP_RESULTS_DIR / "MIST/val_pretrained/good/fake_B",
    "Norm": PSP_RESULTS_DIR / "MIST/val_pretrained/norm/fake_B",
}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def natural_key(value: str):
    """Sort names naturally."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


@st.cache_data(show_spinner=False)
def list_image_names(folder_string: str) -> tuple[str, ...]:
    folder = Path(folder_string)

    if not folder.is_dir():
        return tuple()

    return tuple(sorted(
        (
            p.name
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_key,
    ))


@st.cache_data(show_spinner=False, max_entries=160)
def make_thumbnail(
    path_string: str,
    mtime_ns: int,
    max_edge: int = 420,
) -> bytes:
    """Create an in-memory PNG preview. Original files are never modified."""
    del mtime_ns

    path = Path(path_string)

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        image.thumbnail(
            (max_edge, max_edge),
            Image.Resampling.LANCZOS,
        )

        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)

        return buffer.getvalue()


def build_source_maps(source_specs: dict, model_name: str):
    """
    Return {source_label: {sample_key: image_path}}.

    Standard models use the complete filename as the key.
    BCI removes the configured suffix (_real_A, _real_B, _fake_B)
    from the filename stem and uses the remaining sample name.
    """
    source_maps = {}
    invalid_files = {}

    for label, spec in source_specs.items():
        folder = spec["folder"]
        suffix = spec.get("suffix")

        file_map = {}
        invalid = []

        for filename in list_image_names(str(folder)):
            path = folder / filename

            if suffix is None:
                sample_key = filename
            else:
                file_stem = path.stem

                if not file_stem.endswith(suffix):
                    continue

                sample_key = file_stem[:-len(suffix)]

                if not sample_key:
                    invalid.append(filename)
                    continue

            if sample_key in file_map:
                invalid.append(filename)
                continue

            file_map[sample_key] = path

        source_maps[label] = file_map

        if invalid:
            invalid_files[label] = invalid

    return source_maps, invalid_files


def get_subset_sets() -> Dict[str, Set[str]]:
    return {
        label: set(list_image_names(str(folder)))
        for label, folder in FILTER_DIRS.items()
    }


def subset_keys_for_model(names: Set[str], filter_mode: str) -> Set[str]:
    """Convert subset filenames to the key format used by the selected model."""
    if filter_mode == "stem":
        return {Path(name).stem for name in names}

    return set(names)


# ---------------------------------------------------------
# Original image dialog
# ---------------------------------------------------------

@st.dialog("Original image", width="large")
def show_original(path_string: str, label: str, sample_key: str):
    path = Path(path_string)

    st.caption(f"{label} · {sample_key}")

    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source)

            st.image(image, use_container_width=True)

            st.caption(
                f"Original file: {path.name} · "
                f"{source.width} × {source.height} px · "
                f"{source.format or 'unknown format'}"
            )

    except Exception as exc:
        st.error(f"Could not open image: {exc}")

    if st.button("Close", use_container_width=True):
        st.session_state.pop("selected_image", None)
        st.rerun()


# ---------------------------------------------------------
# Sidebar: model selection
# ---------------------------------------------------------

st.title("Image Generation Comparison")

st.sidebar.header("Display")

selected_model = st.sidebar.selectbox(
    "Select model",
    list(MODEL_CONFIGS.keys()),
    index=0,
)

config = MODEL_CONFIGS[selected_model]
SOURCE_SPECS = config["source_specs"]
REFERENCE = config["reference"]
RESULTS_DIR = config["results_dir"]
FILTER_MODE = config["filter_mode"]


# ---------------------------------------------------------
# Validate directories
# ---------------------------------------------------------

missing_dirs = sorted({
    str(spec["folder"])
    for spec in SOURCE_SPECS.values()
    if not spec["folder"].is_dir()
})

if missing_dirs:
    st.error(
        f"Some required image folders for {selected_model} "
        "were not found:"
    )

    for path in missing_dirs:
        st.code(path)

    st.info(f"Results directory: {RESULTS_DIR}")
    st.stop()


# ---------------------------------------------------------
# Load image names and compare datasets
# ---------------------------------------------------------

source_maps, invalid_files = build_source_maps(
    SOURCE_SPECS,
    selected_model,
)

source_sets = {
    label: set(file_map.keys())
    for label, file_map in source_maps.items()
}

reference_names = source_sets[REFERENCE]

if not reference_names:
    st.error(
        f"No images found in reference source '{REFERENCE}' "
        f"({SOURCE_SPECS[REFERENCE]['folder']})."
    )
    st.stop()


mismatch_messages = []

for label, names in source_sets.items():
    if names != reference_names:
        missing = sorted(
            reference_names - names,
            key=natural_key,
        )

        extra = sorted(
            names - reference_names,
            key=natural_key,
        )

        mismatch_messages.append((label, missing, extra))


if invalid_files:
    st.warning(
        "Some files were skipped because their names did not match "
        "the expected pattern or because multiple files mapped to "
        "the same sample name."
    )

    for label, filenames in invalid_files.items():
        with st.expander(f"{label}: {len(filenames)} skipped files"):
            st.code("\n".join(filenames[:30]))


if mismatch_messages:
    st.warning(
        "The image sets are not identical. "
        "The table will show only samples present in every source."
    )

    for label, missing, extra in mismatch_messages:
        with st.expander(
            f"{label}: {len(missing)} missing, {len(extra)} extra"
        ):
            if missing:
                st.write(f"Missing relative to {REFERENCE}:")
                st.code("\n".join(missing[:30]))

            if extra:
                st.write("Extra:")
                st.code("\n".join(extra[:30]))


all_names = set.intersection(*source_sets.values())


if not all_names:
    st.error("No common image samples found across all sources.")
    st.stop()


# ---------------------------------------------------------
# Load shared subsets
# ---------------------------------------------------------

subset_sets = get_subset_sets()

comb_subset_keys = subset_keys_for_model(
    subset_sets["Comb"],
    FILTER_MODE,
)
good_subset_keys = subset_keys_for_model(
    subset_sets["Good"],
    FILTER_MODE,
)
norm_subset_keys = subset_keys_for_model(
    subset_sets["Norm"],
    FILTER_MODE,
)

comb_names = comb_subset_keys & all_names
good_names = good_subset_keys & all_names
norm_names = norm_subset_keys & all_names

if not subset_sets["Comb"]:
    st.warning(
        "PSPStain MIST comb/fake_B is empty or unavailable. "
        "Comb-related filters may be empty."
    )

if (
    (subset_sets["Good"] | subset_sets["Norm"])
    != subset_sets["Comb"]
    or (subset_sets["Good"] & subset_sets["Norm"])
):
    st.warning(
        "Good and Norm do not form a disjoint partition of Comb "
        "in the PSPStain subset directories."
    )


# ---------------------------------------------------------
# Sidebar: filters and pagination
# ---------------------------------------------------------

filter_options = {
    f"All ({len(all_names)})": "All",
    f"Comb ({len(comb_names)})": "Comb",
    f"Good ({len(good_names)})": "Good",
    f"Norm ({len(norm_names)})": "Norm",
    f"Other (not in comb) ({len(all_names - comb_names)})": "Other",
}

selected_filter_label = st.sidebar.selectbox(
    "Example set",
    list(filter_options.keys()),
)

selected_filter = filter_options[selected_filter_label]

page_size = st.sidebar.selectbox(
    "Rows per page",
    [5, 10, 20],
    index=1,
)


if selected_filter == "All":
    selected_names = all_names
elif selected_filter == "Comb":
    selected_names = comb_names
elif selected_filter == "Good":
    selected_names = good_names
elif selected_filter == "Norm":
    selected_names = norm_names
else:
    selected_names = all_names - comb_names


ordered_names = sorted(selected_names, key=natural_key)

total = len(ordered_names)

pages = max(
    1,
    (total + page_size - 1) // page_size,
)

page = st.sidebar.number_input(
    "Page",
    min_value=1,
    max_value=pages,
    value=1,
    step=1,
    key=f"page_{selected_model}_{selected_filter}_{page_size}",
)

start = (page - 1) * page_size
page_names = ordered_names[start:start + page_size]


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

st.subheader(selected_model)
st.caption(f"Results directory: {RESULTS_DIR}")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Available in all methods",
    len(all_names),
)

c2.metric(
    "Filtered (Comb)",
    len(comb_names),
)

c3.metric(
    "Examples in current filter",
    total,
)


st.write(
    f"**{selected_filter_label}** · "
    f"examples {start + 1 if total else 0}–"
    f"{min(start + page_size, total)} of {total}"
)


# ---------------------------------------------------------
# Image comparison table
# ---------------------------------------------------------

if not page_names:
    st.info("No examples match this filter.")

else:
    labels = list(SOURCE_SPECS.keys())

    header_cols = st.columns(
        [1] * len(labels),
        gap="small",
    )

    for col, label in zip(header_cols, labels):
        col.markdown(f"**{label}**")

    for row_number, sample_key in enumerate(
        page_names,
        start=start + 1,
    ):
        cols = st.columns(
            [1] * len(labels),
            gap="small",
        )

        for col, label in zip(cols, labels):
            image_path = source_maps[label][sample_key]

            try:
                stat = image_path.stat()

                preview = make_thumbnail(
                    str(image_path),
                    stat.st_mtime_ns,
                )

                col.image(
                    preview,
                    use_container_width=True,
                )

                col.button(
                    "Open original",
                    key=f"open_{selected_model}_{label}_{sample_key}",
                    use_container_width=True,
                    on_click=lambda p=str(image_path), l=label, k=sample_key: (
                        st.session_state.update(
                            selected_image=(p, l, k)
                        )
                    ),
                )

            except Exception as exc:
                col.error(
                    f"Cannot display\n{image_path.name}\n{exc}"
                )

        st.caption(f"{row_number}. {sample_key}")
        st.divider()


# ---------------------------------------------------------
# Original image viewer
# ---------------------------------------------------------

if "selected_image" in st.session_state:
    selected_path, selected_label, selected_sample_key = (
        st.session_state["selected_image"]
    )

    show_original(
        selected_path,
        selected_label,
        selected_sample_key,
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "Previews are generated in memory as PNG. "
    "Original image files are read-only and are not "
    "recompressed or overwritten."
)
