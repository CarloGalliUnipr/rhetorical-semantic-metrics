# Reproducibility notes

Exact regeneration of LLM labels is not guaranteed because the original annotation calls did not fix all decoding parameters. The processed sentence-level labels and confidence values should therefore be treated as the reproducible record of the study.

The code supports rerunning the method on new data, but new LLM calls may differ from the frozen outputs.

Full abstract and sentence text are not redistributed in this repository. Users can rehydrate records from PubMed using PMIDs or run the method on their own abstracts.
