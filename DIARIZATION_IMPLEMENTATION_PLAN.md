# Diarization for Uploaded Recordings — Implementation Plan

## Goal

Add a backend-driven pipeline that ingests a pre-recorded audio file, runs
**Whisper transcription** and **pyannote speaker diarization**, aligns them, and
exposes the diarized transcript through the existing meeting API so the
frontend can display speaker-labeled transcripts.

Scope: **uploads only** (existing recordings). Live-recording diarization is
out of scope for this plan.

**Primary target language: Russian (ru).** All transcription and language-
sensitive choices below assume Russian audio. pyannote diarization itself is
language-agnostic, but Whisper transcription must be configured for Russian
(see "Russian language configuration" below).

## Architecture

```
Frontend "Upload recording" button
        │  multipart POST
        ▼
POST /api/meetings/upload  ──► save file, create meeting row (status=processing)
        │
        │ (BackgroundTasks)
        ▼
ingestion.process(meeting_id, audio_path)
   ├── 1. normalize_audio()   ── pydub/ffmpeg → 16kHz mono WAV
   ├── 2. transcribe()        ── whisper-server :8178 /inference
   ├── 3. diarize()           ── pyannote/speaker-diarization-3.1
   ├── 4. align_speakers()    ── max-overlap matching by timestamp
   └── 5. persist()           ── transcripts rows w/ speaker, status=done
        │
        ▼
GET /api/meetings/{id}/status      ── polled by frontend
GET /api/meetings/{id}             ── returns transcripts with speaker field
```

Pyannote runs **in-process** in the FastAPI backend (per user decision). Model
loaded once at app startup and reused across requests.

## Dependency additions

Append to `backend/requirements.txt`:

```
pyannote.audio==3.3.2
torch>=2.2
torchaudio>=2.2
pydub==0.25.1
requests==2.32.3
```

Notes:
- Installs ~2GB. Acceptable per project decision to keep diarization in the
  main backend.
- Document `HF_TOKEN` env var in `.env.example` — required to download the
  pyannote model (user must accept license at
  https://hf.co/pyannote/speaker-diarization-3.1).
- `ffmpeg` binary must be on PATH for pydub format conversion. Document in
  README per-OS (macOS: `brew install ffmpeg`, Windows: winget/choco, Linux:
  `apt install ffmpeg`).

## Schema changes (`backend/app/db.py`)

### `meetings` table — add columns

| Column              | Type    | Default        | Notes                                  |
|---------------------|---------|----------------|----------------------------------------|
| `audio_file`        | TEXT    | NULL           | Absolute path to stored upload WAV     |
| `processing_status` | TEXT    | `'done'`       | `processing` \| `done` \| `failed`     |
| `processing_error`  | TEXT    | NULL           | Last failure message if any            |

### `transcripts` table — add column

| Column    | Type | Default | Notes                                       |
|-----------|------|---------|---------------------------------------------|
| `speaker` | TEXT | NULL    | e.g. `SPEAKER_00`, renameable in frontend   |

### Migration

Add an idempotent migration helper in `db.py` (runs on `DatabaseManager`
init):

```python
async def _migrate(self):
    for stmt in [
        "ALTER TABLE meetings ADD COLUMN audio_file TEXT",
        "ALTER TABLE meetings ADD COLUMN processing_status TEXT DEFAULT 'done'",
        "ALTER TABLE meetings ADD COLUMN processing_error TEXT",
        "ALTER TABLE transcripts ADD COLUMN speaker TEXT",
    ]:
        try:
            await db.execute(stmt)
        except aiosqlite.OperationalError:
            pass  # column already exists
```

Update the `Transcript` Pydantic model in `main.py`:

```python
class Transcript(BaseModel):
    id: str
    text: str
    timestamp: str
    audio_start_time: Optional[float] = None
    audio_end_time: Optional[float] = None
    duration: Optional[float] = None
    speaker: Optional[str] = None      # NEW
```

## New module: `backend/app/diarization.py`

Responsibilities:
- Lazy-load pyannote pipeline once.
- Run diarization inside `asyncio.to_thread` so it doesn't block the loop.
- Return a list of speaker turns.

```python
from dataclasses import dataclass
from functools import lru_cache
import os, asyncio, logging
import torch
from pyannote.audio import Pipeline

log = logging.getLogger(__name__)

@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str

@lru_cache(maxsize=1)
def _pipeline() -> Pipeline:
    token = os.environ["HF_TOKEN"]
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )
    if torch.cuda.is_available():
        pipe.to(torch.device("cuda"))
        log.info("pyannote running on CUDA")
    else:
        log.info("pyannote running on CPU (slow)")
    return pipe

def _run(wav_path: str) -> list[SpeakerTurn]:
    diarization = _pipeline()(wav_path)
    return [
        SpeakerTurn(start=t.start, end=t.end, speaker=spk)
        for t, _, spk in diarization.itertracks(yield_label=True)
    ]

async def diarize(wav_path: str) -> list[SpeakerTurn]:
    return await asyncio.to_thread(_run, wav_path)
```

## New module: `backend/app/ingestion.py`

```python
import os, uuid, logging, subprocess, requests
from pathlib import Path
from pydub import AudioSegment
from diarization import diarize, SpeakerTurn

log = logging.getLogger(__name__)

UPLOAD_DIR = Path(os.environ.get("MEETILY_UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
WHISPER_URL = os.environ.get("WHISPER_URL", "http://localhost:8178/inference")

def normalize_audio(src: Path) -> Path:
    """Convert any input to 16kHz mono PCM WAV (Whisper + pyannote happy)."""
    out = src.with_suffix(".norm.wav")
    AudioSegment.from_file(src).set_frame_rate(16000).set_channels(1).export(
        out, format="wav"
    )
    return out

def transcribe(wav: Path) -> list[dict]:
    """Returns [{start, end, text}, ...] from whisper-server."""
    with open(wav, "rb") as f:
        r = requests.post(
            WHISPER_URL,
            files={"file": f},
            data={"response_format": "verbose_json", "temperature": "0.0"},
            timeout=60 * 60,
        )
    r.raise_for_status()
    return r.json()["segments"]

def align_speakers(
    segments: list[dict], turns: list[SpeakerTurn]
) -> list[dict]:
    """Attach speaker label by max temporal overlap."""
    for seg in segments:
        s, e = seg["start"], seg["end"]
        best, best_ov = None, 0.0
        for t in turns:
            ov = max(0.0, min(e, t.end) - max(s, t.start))
            if ov > best_ov:
                best, best_ov = t.speaker, ov
        seg["speaker"] = best
    return segments

async def process(db, meeting_id: str, audio_path: Path):
    try:
        norm = normalize_audio(audio_path)
        segments = transcribe(norm)
        turns = await diarize(str(norm))
        merged = align_speakers(segments, turns)
        await db.replace_transcripts(meeting_id, merged)
        await db.set_meeting_status(meeting_id, "done")
    except Exception as e:
        log.exception("ingestion failed")
        await db.set_meeting_status(meeting_id, "failed", error=str(e))
```

## API surface (`backend/app/main.py`)

### `POST /api/meetings/upload`

```python
from fastapi import UploadFile, File, Form, BackgroundTasks

@app.post("/api/meetings/upload")
async def upload_meeting(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form("Untitled upload"),
):
    meeting_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{meeting_id}{Path(file.filename).suffix}"
    with open(dest, "wb") as out:
        out.write(await file.read())
    await db.create_meeting(meeting_id, title, audio_file=str(dest),
                            processing_status="processing")
    background.add_task(ingestion.process, db, meeting_id, dest)
    return {"meeting_id": meeting_id, "status": "processing"}
```

### `GET /api/meetings/{id}/status`

```python
@app.get("/api/meetings/{meeting_id}/status")
async def get_status(meeting_id: str):
    row = await db.get_meeting_status(meeting_id)
    if not row:
        raise HTTPException(404)
    return row  # {status, error?}
```

Existing `GET /api/meetings/{id}` requires no change beyond returning the
new `speaker` field via the updated Pydantic model.

### DB helpers to add (`db.py`)

- `create_meeting(id, title, audio_file=None, processing_status='done')`
- `set_meeting_status(id, status, error=None)`
- `get_meeting_status(id) -> {status, error}`
- `replace_transcripts(meeting_id, segments)` — wipes any prior transcripts
  for the meeting and inserts the new aligned segments with `speaker`,
  `audio_start_time`, `audio_end_time`, `duration`, `text`, generated UUID id.

## Frontend changes

**Location:** [frontend/src/components/Sidebar/](frontend/src/components/Sidebar/)
and [frontend/src/app/page.tsx](frontend/src/app/page.tsx).

1. **Upload button** in the sidebar / new-meeting menu:
   - Opens native file picker (Tauri `@tauri-apps/plugin-dialog`).
   - POSTs the file to `/api/meetings/upload` via `fetch` with `FormData`.
   - Stores returned `meeting_id`, navigates to that meeting.

2. **Processing state**:
   - When the opened meeting has `processing_status === 'processing'`, show a
     spinner + "Transcribing and identifying speakers…".
   - Poll `GET /api/meetings/{id}/status` every 3s until `done` or `failed`.
   - On `done`, refetch the meeting and render transcripts.

3. **Speaker labels in transcript view**:
   - Render `transcript.speaker` as a colored chip before the text.
   - Maintain a per-meeting `speakerNames` map in local state / localStorage,
     so users can rename `SPEAKER_00 → "Alice"`. Renames are display-only for
     v1 (no backend persistence). Persistence can be added later via a
     `speaker_names` JSON column on `meetings`.

4. **Type updates** in the frontend transcript type so `speaker?: string`
   flows through.

## Environment configuration

Document in `backend/.env.example` (create if absent):

```
HF_TOKEN=hf_xxx                          # required for pyannote
WHISPER_URL=http://localhost:8178/inference
MEETILY_UPLOAD_DIR=./uploads
MEETILY_TRANSCRIBE_LANGUAGE=ru           # forces Whisper to Russian
```

## Russian language configuration

### Whisper

- **Model choice**: avoid `*.en` models (English-only). Use multilingual
  builds. Recommended minimum **`medium`** for Russian; **`large-v3`** or
  **`large-v3-turbo`** strongly preferred — small/base produce noticeably
  worse Russian WER (often 20%+ vs ~8–10% on large-v3).
- **Build**: rerun `./build_whisper.sh medium` (or `large-v3-turbo`) on
  macOS / `build_whisper.cmd` on Windows so the whisper-server hosts a
  Russian-capable model.
- **Force language** in the `/inference` call to prevent auto-detect drift
  on short or noisy clips:

  ```python
  r = requests.post(
      WHISPER_URL,
      files={"file": f},
      data={
          "response_format": "verbose_json",
          "temperature": "0.0",
          "language": "ru",          # force Russian
          "task": "transcribe",       # not "translate"
      },
      timeout=60 * 60,
  )
  ```

- Make the language configurable via env var `MEETILY_TRANSCRIBE_LANGUAGE`
  (default `"ru"`) so this stays flexible.

### pyannote

No Russian-specific config needed — speaker-diarization-3.1 uses acoustic
features (not lexical), so it works on any language. Verified by pyannote
maintainers on multilingual benchmarks.

### Alignment & display

- Whisper segments for Russian sometimes split mid-word on hesitations; the
  max-overlap aligner handles this correctly because it operates on time,
  not text.
- Ensure DB and JSON responses are UTF-8 end-to-end (SQLite is by default;
  FastAPI is by default). No extra work expected, but verify with a sample
  Russian transcript that Cyrillic round-trips through
  `GET /api/meetings/{id}`.

### LLM summarization (downstream, already in `transcript_processor.py`)

Out of scope for this plan, but flag: if summaries are generated from
diarized Russian transcripts, the prompt template should instruct the model
to summarize **in Russian**. Track as a follow-up.

## Implementation order

1. **DB migration + Pydantic model** (`db.py`, `main.py`). Smallest change,
   verifiable with the existing API still working.
2. **`diarization.py`** + standalone smoke test against a known WAV. Confirm
   model download + GPU/CPU selection works.
3. **`ingestion.py`** + manual run against a known WAV with whisper-server
   already running. Verify alignment output looks right.
4. **Upload + status endpoints** wired to ingestion as a background task.
   Test with `curl -F file=@sample.wav`.
5. **Frontend**: upload button → status polling → speaker-labeled rendering.
6. **Speaker rename UI** (display-only, localStorage).
7. **README updates**: HF token setup, ffmpeg install, expected processing
   times.

## Testing strategy

- **Unit**: `align_speakers` with synthetic segments/turns — verify
  max-overlap correctness, empty-turns fallback (speaker=None), single-turn
  span covering multiple segments.
- **Integration**: place a short multi-speaker WAV in
  `backend/tests/fixtures/`, run `ingestion.process` against a live
  whisper-server, assert transcript count > 0 and at least 2 distinct
  speakers detected.
- **Manual**: upload a real 5-minute 2-speaker recording end-to-end, confirm
  UI renders correctly.

## Known caveats / risks

- **CPU diarization is slow** (~0.3× realtime). A 1-hour recording = ~3h on
  CPU. Document this; encourage GPU torch install on Windows/Linux. macOS
  pyannote on MPS is supported in 3.1 but quality varies.
- **HF token gating**: first run requires manual license acceptance on
  HuggingFace. Surface a clear error message if `HF_TOKEN` missing or model
  not accessible.
- **whisper-server dependency**: ingestion fails fast if it's not running.
  Future: optional fallback to `whisper-cli` subprocess.
- **Memory**: pyannote loads ~500MB into RAM/VRAM. Document min RAM (8GB).
- **Concurrency**: pyannote pipeline is not thread-safe across concurrent
  diarize calls. Guard with an `asyncio.Lock` inside `diarization.py` if
  multiple uploads can overlap.
- **Large files**: `await file.read()` loads the entire upload in memory.
  Switch to streaming write (`shutil.copyfileobj` on `file.file`) before
  shipping if uploads > a few hundred MB are expected.

## Out of scope (future work)

- Diarizing live recordings produced by the Tauri frontend.
- Persisting custom speaker names server-side.
- Speaker embedding-based identification across meetings ("this is Alice
  again").
- Re-diarization on demand for an existing meeting.
