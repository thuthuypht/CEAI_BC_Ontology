paper_assets/explanations
========================
This folder contains reproducibility artifacts for the SPARQL-based semantic reasoning component.

Detected namespaces
------------------
- WDBC:    http://example.org/breastcancer#
- Coimbra: http://example.org/coimbra#

Files
-----
- prefixes.sparql
  Prefix declarations for both datasets (wdbc:, coim:). The generic bc: prefix aliases WDBC by default.
- prefixes_wdbc.sparql / prefixes_coimbra.sparql
  Convenience prefix files where bc: aliases the dataset namespace.
- sparql_templates.txt
  Reusable SPARQL templates for (i) rule materialization and (ii) cluster explanations.
- semantic_feature_map.csv / semantic_feature_map.json
  Mapping from each rule output to feature columns used in ML.
- example_outputs/
  Example exports produced by the pipeline (e.g., cluster explanations).

Thresholds
----------
If thresholds are data-driven, compute them on the training split only (e.g., percentiles),
then fix the numeric values in the SPARQL filters. Report final values in Appendix Table A1.
