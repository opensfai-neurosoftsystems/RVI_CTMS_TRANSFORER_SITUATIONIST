# Multi-Matrix Corpus Record

Training documents stay intact outside this repository. The training layer stores
passage-level records with provenance and operator/matrix labels.

```json
{
  "id": "TD-001:000001",
  "text": "passage text",
  "matrices": ["semantic", "temporal", "contradiction"],
  "operator": "REDCHECK",
  "metadata": {
    "document_id": "TD-001",
    "title": "source title",
    "page": 1,
    "source_hash": "..."
  }
}
```

The text is not replaced by the labels. The labels create alternate addresses around it.
