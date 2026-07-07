# MUD Project Documentation Index

This directory contains all the architectural, research, and audit documentation for the Forge LLM (MUD) project.

## Directory Structure

*   **[`audits/`](audits/)**: Sequential audit reports documenting the evolution, debugging, and validation of the engine. `MUD_AUDIT_LATEST.md` = V34 (JEPA Collapse & Residual Scaling).
*   **[`sessions/`](sessions/)**: Chronological session reports detailing daily accomplishments. Latest: `MUD_SESSION_REPORT_2026-07-01.md` (JEPA Gate Rewire, Telemetry, P-13 Audit).
*   **[`architecture/`](architecture/)**: Specifications, manifestos, hardware ISA/kernel plans, and deep dives into core components.
*   **[`research/`](research/)**: Notes on academic papers, external implementations, and theoretical research.
*   **[`manuals/`](manuals/)**: User manuals, operational protocols, calibration procedures, roadmaps, and guidelines.
*   **[`dumps/`](dumps/)**: Raw text dumps, debug logs, and disassembly outputs.
*   **[`dumps_archive/`](dumps_archive/)**: Archived dumps from previous sessions (historical reference only).
*   **[`logs_archive/`](logs_archive/)**: Archived log files from previous runs (historical reference only).

## Key Documents to Start With

1.  **[MUD Overview](architecture/MUD_OVERVIEW.md)**: High-level summary of the engine's capabilities and goals.
2.  **[Engine Manifesto](architecture/ENGINE_MANIFESTO.md)**: Core philosophical and technical mandates governing the codebase.
3.  **[MUD User Manual](manuals/MUD_USER_MANUAL.md)**: Practical instructions for converting, calibrating, and training models.
4.  **[Master Roadmap](manuals/MUD_ROADMAP_MASTER.md)**: The consolidated master plan merging all roadmaps, kernel plans, and upgrade paths.
5.  **[UCP v2 Protocol](manuals/MUD_UNIVERSAL_PROTOCOL_V2.md)**: Universal Calibration Protocol — required steps for every new model.
6.  **[Latest Audit](audits/MUD_AUDIT_LATEST.md)**: Current technical status (V34 — JEPA Collapse & Residual Scaling).
7.  **[AGENTS.md](../AGENTS.md)**: Project context for AI agents — most up-to-date technical reference.
8.  **[JEPA Doppler Filter](research/JEPA_DOPPLER_FILTER.md)**: JEPA gate theory, centering, and soft-clipping mechanics.
