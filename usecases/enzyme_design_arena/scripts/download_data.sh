
#!/usr/bin/env bash
set -euo pipefail

ROOT="usecases/enzyme_design_arena/data/raw"
mkdir -p "$ROOT"/{nucb,modify_rma,mcba,proteingym_optional}
mkdir -p usecases/enzyme_design_arena/data/{processed,manifests}

curl -L \
  https://raw.githubusercontent.com/google-deepmind/nuclease_design/main/data/landscape.csv \
  -o "$ROOT/nucb/landscape.csv"

curl -L \
  "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-50698-y/MediaObjects/41467_2024_50698_MOESM1_ESM.pdf" \
  -o "$ROOT/modify_rma/supplement.pdf"

curl -L \
  "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-50698-y/MediaObjects/41467_2024_50698_MOESM4_ESM.xlsx" \
  -o "$ROOT/modify_rma/supplementary_data_1.xlsx"

if [ ! -d "$ROOT/mcba/accelerated_enzyme_engineering_repo/.git" ]; then
  rm -rf "$ROOT/mcba/accelerated_enzyme_engineering_repo"
  git clone https://github.com/grantlandwehr/accelerated-enzyme-engineering.git \
    "$ROOT/mcba/accelerated_enzyme_engineering_repo"
fi

curl -L \
  "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-55399-0/MediaObjects/41467_2024_55399_MOESM4_ESM.xlsx" \
  -o "$ROOT/mcba/source_data.xlsx"

curl -L \
  "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-55399-0/MediaObjects/41467_2024_55399_MOESM1_ESM.pdf" \
  -o "$ROOT/mcba/supplement.pdf"

curl -L \
  https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_substitutions.csv \
  -o "$ROOT/proteingym_optional/DMS_substitutions.csv"

cat > "$ROOT/nucb/README_source.txt" <<'TXT'
Source: google-deepmind/nuclease_design
File: data/landscape.csv
Purpose: NucB sequence-activity landscape benchmark
TXT

cat > "$ROOT/modify_rma/README_source.txt" <<'TXT'
Source article DOI: 10.1038/s41467-024-50698-y
Files:
- supplement.pdf
- supplementary_data_1.xlsx
Purpose: Rma cytochrome c library/task pack for C-B and C-Si reactions
TXT

cat > "$ROOT/mcba/README_source.txt" <<'TXT'
Source article DOI: 10.1038/s41467-024-55399-0
Associated repo: grantlandwehr/accelerated-enzyme-engineering
Files:
- source_data.xlsx
- supplement.pdf
- accelerated_enzyme_engineering_repo/
Purpose: McbA task pack for reaction-conditioned enzyme engineering
TXT

cat > "$ROOT/proteingym_optional/README_source.txt" <<'TXT'
Source: ProteinGym v1.3
File: DMS_substitutions.csv
Purpose: optional extension benchmark for mutation-effect ranking
TXT
