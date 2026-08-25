# Downstream pipeline: architecture and end-to-end demo

Phase one, 2026-07-06. Status at the time of writing: M1 to M5 all working, 44 tests with 2
skipped. This is a snapshot of that phase; the suite has grown since.

This phase built the second half of the system — perceive, remember, recall — into something
that demonstrates every function while remaining pure software plus mocks. No real hardware and
no real API calls.

## 1. Data flow

All of this runs.

```
gatekeeper fires (mock signal + confidence)      MockGatekeeper.trigger()
        |
   grab full-res frame (test image stands in)    capture.grab_frame()  -> data/mvp_demo/frames/
        |
   local OCR                                     OCRInterface -> TesseractOCR (real) / StubOCR (tests)
        |  ocr_text
   package {ocr_text + metadata}                 packaging.build_payload()
        |  (timestamp / trigger_confidence / raw_image_policy)
   +----+------------ is_online() mock ------------+
   | online                                        | offline
   transport.upload -> enrich (mock tags)          store as pending, tags empty
   -> status=done, store                           IngestService._queue()
        |                                              | connectivity returns
        |                                         process_pending() backfills tags -> done
   privacy: raw_image_policy handles the frame    privacy.apply_raw_image_policy()
   (delete by default; cache optional)
        |
   SQLite storage                                 db.CardStore -> data/mvp_demo/memosight.db
        |
   command-line recall                            cli.py: list / show / search / pending
```

## 2. Three swappable interfaces

These are what keep the system extensible.

| Interface | Abstract base | Implementation in this phase | Future replacement |
|------|----------|-----------|----------|
| OCR | `pipeline/ocr/base.py::OCRInterface` | `TesseractOCR` (real engine), `StubOCR` (tests) | On-phone or cloud OCR |
| Enricher | `pipeline/enrich/base.py::EnricherInterface` | `CloudEnricher`, mocked, returning `mock:`-prefixed fake tags | A real Claude API call |
| Transport / upload | `pipeline/transport/base.py::UploadInterface` | `DirectUploadMock`, a mock of the Pi uploading directly | Relay via a phone |

Tags are the only field produced by a cloud model. In this phase their only source is the
mocked `CloudEnricher`; there is no rule-based implementation. When offline, the pipeline never
fabricates tags from rules — it queues the card and waits for the real cloud call. That way a
card's tags always mean the same thing.

## 3. Modules

```
pipeline/
  models.py        MemoryCard data model, matching the fixed SQLite table
  db.py            CardStore: schema, CRUD, pending queue, search
  config.py        Config: raw_image_policy (delete by default), directories, OCR language
  packaging.py     Payload: ocr_text plus metadata
  connectivity.py  Connectivity ABC and ConnectivityMock, with switchable is_online
  capture.py       MockGatekeeper (mock trigger) and grab_frame
  privacy.py       apply_raw_image_policy (delete / cache)
  ingest.py        IngestService: store directly when online, queue when offline, backfill on recovery
  pipeline.py      MemoSightPipeline orchestration plus the build_pipeline() factory
  cli.py           Command-line recall and demo
  ocr/             OCR interface, Tesseract, Stub
  enrich/          Enricher interface, CloudEnricher (mock)
  transport/       Transport interface, DirectUploadMock
tests/             44 unittest cases, stdlib only, no pytest needed
```

## 4. Running it

```bash
# All tests. Without the tesseract binary, 2 real-engine tests skip
.venv/bin/python -m unittest discover -s tests -v

# Built-in end-to-end demo: synthesise a text image, run the full loop, search it back out
.venv/bin/python -m pipeline.cli demo

# Ingest one image, real or synthetic
.venv/bin/python -m pipeline.cli ingest <image.png> --confidence 0.9
.venv/bin/python -m pipeline.cli ingest <image.png> --offline    # simulate being offline
.venv/bin/python -m pipeline.cli process-pending                 # backfill once online

# Recall
.venv/bin/python -m pipeline.cli list
.venv/bin/python -m pipeline.cli search <keyword>
.venv/bin/python -m pipeline.cli show <id>
.venv/bin/python -m pipeline.cli pending

# The database path can be overridden: MEMOSIGHT_DB=/path/to.db
```

## 5. What is mocked in this phase

OCR engine: Tesseract, chosen 2026-07-06. `pytesseract` and `pillow` are installed in the venv;
the system binary has to be installed separately:

```
sudo apt install -y tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng
```

Once installed, the 2 skips in `test_ocr.py` become real runs and `cli demo` produces real
recognition ("MEMOSIGHT DEMO ROADMAP") instead of placeholder text. Without the binary the
pipeline falls back to `StubOCR` and the loop still demonstrates end to end.

Enricher: mocked `CloudEnricher`, returning fake tags with a `mock:` prefix so they are
obviously not real. Phase two swaps in a real Claude API call and a prompt. The key
infrastructure is already in place: `pipeline/env.py::load_env()` uses python-dotenv to load
`ANTHROPIC_API_KEY` from a `.env` at the project root, so it does not depend on a shell export
and any process can read it, and `get/require_anthropic_api_key()` reads from `os.environ`.
Keys are never hardcoded and `.env` is gitignored. Both the CLI entry point and
`build_pipeline()` call it.

Transport: `DirectUploadMock`, a stand-in for the Pi uploading directly. Phase two can add a
phone-relay implementation.

Gatekeeper and camera: `MockGatekeeper` plus test images. Phase two connects the real gatekeeper
(task1 C_wide_uniform int8) and a real camera.

## 6. Test coverage

44 tests, 2 skipped.

- `test_db.py` (14): model validation, CRUD, search, enrichment state transitions, FIFO pending,
  persistence across reopen
- `test_ocr.py` (10, 2 skipped): preprocessing resize and greyscale, StubOCR, the real Tesseract
  engine (pending the binary)
- `test_enrich.py` (9): interface, mock tag shape, reproducibility, confidence tagging, simulated
  failure, packaging
- `test_queue.py` (6): direct store when online, queue when offline, backfill on recovery, cloud
  failure fallback, failures stay pending
- `test_e2e.py` (5): online capture and recall, no record when not triggered, offline then
  recovery, frame deleted by default, frame retained under cache

## 7. Backlog and the entry point to phase two

- Install the tesseract binary (section 5), then re-check real-engine quality on Chinese and
  English.
- Review this phase and decide whether to proceed to phase two: a real Claude API enricher, the
  real gatekeeper and camera integrated, and phone relay.
