# compound-sdk

Records the shape of every model call your app makes: token counts, timing,
cache hits, and a salted hash of each prompt prefix. It never keeps the text of
a prompt or an answer. The records are what `compound shape` and
`compound estimate` read to tell you what your traffic would cost, and how it
would behave, on other models and hosts.

It has no dependencies and never sits in the request path. Records are written
by a background thread; if anything goes wrong while recording, your app does
not see it.

```python
from openai import OpenAI
from compound_sdk import Recorder

recorder = Recorder(task_key="support")      # writes ~/.compound/shapes.jsonl
client = recorder.wrap_openai(OpenAI())       # every chat.completions call is recorded
```

Any other client: call `recorder.record(messages=..., model=..., started_at=..., usage=...)`
after the call returns.

LiteLLM (SDK or proxy):

```python
import litellm
from compound_sdk.litellm import compound_logger

litellm.callbacks = [compound_logger()]
```

Set `metadata={"task_key": "..."}` on LiteLLM requests to group calls by task.

## Privacy

Prefix hashes are salted with a random secret stored in `~/.compound/salt`,
readable only by your user. Hashes from one install compare with each other,
which is what lets the profile see repeated prompts, but they cannot be
reversed by hashing guesses at a prompt without that file. Do not share it.

Then:

```bash
compound shape ~/.compound/shapes.jsonl --out profile.json
compound estimate ~/.compound/shapes.jsonl --current <your model> --candidates <models>
```
