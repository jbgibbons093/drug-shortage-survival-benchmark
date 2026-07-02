# Data Availability

This repository contains analysis code and aggregate manuscript exhibits. It does not include raw or record-level input data.

The pipeline uses a mix of public administrative sources and licensed data:

- Public sources include FDA National Drug Code Directory, Orange Book, openFDA recalls and adverse events, FDA inspections and warning letters, CMS pricing and utilization files, ASP pricing files, Drug Master Files, FDA establishment registration data, patent and exclusivity data, and EM-DAT disaster data.
- Shortage outcomes are derived from ASHP/UUDIS shortage records used under the authors' data-use terms.
- Licensed prescription features are derived from Symphony Health data and cannot be redistributed.

The scripts in `Programs/` document the expected file names and transformations. A full rerun requires obtaining the licensed and public source files independently and placing them in the local directory structure described by `Programs/00_utilities.py`.

Aggregate tables and figures generated for the manuscript are included under `Manuscripts/generated/tables/` and `Manuscripts/generated/figures/`. These are derived summary artifacts, not raw source data.
