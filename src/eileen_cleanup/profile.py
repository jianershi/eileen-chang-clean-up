# SPDX-License-Identifier: MIT
"""Public source identifiers and the default restoration settings."""

SAMPLE_RATE = 44_100
ENSEMBLE_AUX_SAMPLE_RATE = 16_000

MODEL_REPOSITORY = "nvidia/NVIDIA-NemotronLabs-RE-USE"
MODEL_REVISION = "31f82f2efad1095034492f236e8dac9239ba713b"
MODEL_WEIGHTS_SHA256 = "06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1"

LAUGHTER_MODEL_REPOSITORY = "nvidia/NVIDIA-NemotronLabs-RE-USE-laughter"
LAUGHTER_MODEL_REVISION = "1b1521dd659608e1e063ada83e0acfe1da473a8b"
LAUGHTER_MODEL_WEIGHTS_SHA256 = "35c2a4e284515275d2e49a8e8cb5cd392ae67a1f1d345de1c12a0202c7b6b610"

PINNED_WEIGHTS_SHA256 = {
    (MODEL_REPOSITORY, MODEL_REVISION): MODEL_WEIGHTS_SHA256,
    (LAUGHTER_MODEL_REPOSITORY, LAUGHTER_MODEL_REVISION): LAUGHTER_MODEL_WEIGHTS_SHA256,
}

ARCHIVE_URL = "https://hollisarchives.lib.harvard.edu/catalog/sch01475_sch01475c00589"
AUDIO_URL = "https://mps.lib.harvard.edu/sds/audio/503996508"
CITATION_GUIDE_URL = "https://guides.library.harvard.edu/schlesinger/citation_guide"

# Preserve the catalog wording supplied for the archival source credit.
SOURCE_NOTICE = (
    "Bunting Institute Seminars. Reyher; Bas-Comain. undatedd.. "
    "Radcliffe College Archives sound recordings collection, 1951-2008, "
    "RG XX, Series 8-9, RA.T-106/nd., Audiotape 01. "
    "Schlesinger Library on the History of Women in America, Harvard Radcliffe Institute."
)

OUTPUT_STEM = "NVIDIA_NemotronLabs_REUSE_full_recording"
TITLE = "Bunting Institute Seminars — NVIDIA NemotronLabs RE-USE restoration"

ENSEMBLE_OUTPUT_STEM = "REUSE_ensemble_full_recording"
ENSEMBLE_TITLE = "Bunting Institute Seminars — Multi-pass RE-USE ensemble restoration"
