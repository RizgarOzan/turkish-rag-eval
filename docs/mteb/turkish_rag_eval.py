"""MTEB task definition for TurkishRagEvalRetrieval.

Drop this into a fork of https://github.com/embeddings-benchmark/mteb at
`mteb/tasks/retrieval/tur/turkish_rag_eval.py` and register it in that
directory's `__init__.py`. See ../mteb/README.md in this repository for the
full submission runbook.

The dataset already ships in the layout mteb's RetrievalDatasetLoader expects,
so no `load_data` override is needed:

    corpus   config, `_id` / `title` / `text`   (the loader renames _id -> id)
    queries  config, `_id` / `text`
    default  config, split `test`, `query-id` / `corpus-id` / `score`
"""

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class TurkishRagEvalRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TurkishRagEvalRetrieval",
        dataset={
            "path": "RizgarOzan/turkish-rag-eval",
            # TODO: replace with the commit sha of the dataset revision this
            # task should pin. Get it from the dataset's "Files and versions"
            # tab, or with:
            #     huggingface_hub.HfApi().dataset_info("RizgarOzan/turkish-rag-eval").sha
            "revision": "REPLACE_WITH_HF_COMMIT_SHA",
        },
        description=(
            "Turkish question-answering retrieval over Turkish Wikipedia health "
            "articles. Each question was written by hand as a paraphrase of the "
            "passage that answers it, never a copy, so lexical retrievers get no "
            "unearned advantage; half the corpus is same-domain distractors."
        ),
        reference="https://github.com/RizgarOzan/turkish-rag-eval",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="ndcg_at_10",
        date=("2026-09-01", "2026-09-21"),
        task_subtypes=["Question answering"],
        domains=["Encyclopaedic", "Medical", "Written"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        prompt=(
            "Bir soru verildiğinde, soruyu cevaplayan Wikipedia pasajını getir."
        ),
        bibtex_citation=r"""
@misc{ozan2026turkishrageval,
  author = {Ozan, Rızgar},
  howpublished = {\url{https://huggingface.co/datasets/RizgarOzan/turkish-rag-eval}},
  note = {Harness: \url{https://github.com/RizgarOzan/turkish-rag-eval}},
  title = {Turkish RAG Eval: a retrieval test set for Turkish Wikipedia},
  year = {2026},
}
""",
        contributed_by="RizgarOzan",
    )
