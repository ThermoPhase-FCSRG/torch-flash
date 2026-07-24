# YAML schema documents

These YAML files are JSON Schema 2020-12 documents for external tooling and
editor integration. Runtime validation is implemented without the optional
`jsonschema` dependency and enforces the same envelope, SI-unit, canonical-name,
and type rules.

Model-specific requirements inside `parameters` are documented in
`docs/parameters.md` and validated by the corresponding constructor. Keeping
the common envelope separate lets a new model kind be added without weakening
the component schema or changing existing parameter-set versions.
