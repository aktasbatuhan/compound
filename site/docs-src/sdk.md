title: Recording SDK
order: 64

# Recording SDK

If you have no logs to export, `compound-sdk` records them for you. It is a
small Python package with no dependencies, in `sdk/python` of this repo. It
writes one [shape](../shape/) record per model call: token counts, timing,
cache hits and salted prompt-prefix hashes, and never the text of a prompt or
an answer.

```python
from openai import OpenAI
from compound_sdk import Recorder

recorder = Recorder(task_key="support")    # writes ~/.compound/shapes.jsonl
client = recorder.wrap_openai(OpenAI())     # records every chat.completions call
```

The wrapped client returns exactly what it returned before, streams included.
A stream is recorded when it ends, and its usage is only known if the request
set `stream_options={"include_usage": True}`. Async clients work the same way.

For LiteLLM, as a library or as the proxy:

```python
import litellm
from compound_sdk.litellm import compound_logger

litellm.callbacks = [compound_logger()]
```

For anything else, call `recorder.record(messages=..., model=...,
started_at=..., usage=...)` after the call returns.

## It stays out of the way

Recording happens after the call, and a background thread writes the file.
If the queue fills up, a record is dropped and counted in `recorder.dropped`.
If building a record fails, it is counted in `recorder.errors`. Your app never
sees an exception from the recorder.

## Privacy

The prefix hashes are salted with a random secret in `~/.compound/salt`,
readable only by your user. Records from one machine compare with each other,
which is how repeated prompts show up in the profile, but nobody without that
file can test a guessed prompt against them. Share `shapes.jsonl`, never
`salt`.

Then read the records like any other input:

```bash
compound shape ~/.compound/shapes.jsonl --out profile.json
compound estimate ~/.compound/shapes.jsonl --current <model in use> --candidates <models>
```
