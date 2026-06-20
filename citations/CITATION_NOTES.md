# Citation Notes

## Dataset Citation

| Field | Value |
| --- | --- |
| Title | Album of the Year Dataset |
| Authors/Maintainers | TODO: Add maintainer name |
| Publisher/Host | TODO: Add host (e.g., Kaggle, GitHub) |
| Persistent Identifier | TODO: Add DOI or stable URL |
| Version/Release Date | TODO: Add version or date |
| License | TODO: Add license (e.g., CC BY 4.0) |

## Versioned Software Citations

| Package | Version | Source | Commit SHA | License |
| --- | --- | --- | --- | --- |
| PyMC | TODO | PyPI | N/A | Apache-2.0 |
| ArviZ | TODO | PyPI | N/A | Apache-2.0 |
| scikit-learn | TODO | PyPI | N/A | BSD-3-Clause |
| pandas | TODO | PyPI | N/A | BSD-3-Clause |
| numpy | TODO | PyPI | N/A | BSD-3-Clause |

**How to reproduce environment:** See `requirements.txt` or `pyproject.toml` for exact dependency versions. Run `pip install -e .` to install with locked versions.

## Verification Log (Crossref API)

Date: 2026-01-16

DOI | Status
--- | ---
10.1093/biomet/63.3.581 | 200
10.1214/06-BA117A | 200
10.1016/j.jmva.2009.04.008 | 200
10.1007/s11222-016-9696-4 | 200
10.1214/20-BA1221 | 200
10.1198/016214506000001437 | 200
10.1002/widm.39 | 200
10.1098/rsta.2015.0202 | 200
10.1214/09-AOS735 | 200
10.1214/17-BA1091 | 200

Verification command (requires curl):
```bash
# Check a single DOI via Crossref API (returns HTTP status code)
curl -s -o /dev/null -w "%{http_code}" "https://api.crossref.org/works/10.1093/biomet/63.3.581"
```
