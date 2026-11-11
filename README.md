# hkm-plugin-embeddings

`EmbeddingPort`: text in, `Vector` out — behind a cache keyed on a **content hash**, so
unchanged text is never embedded twice.

Solves `embedding.generation`. Two drivers:

| driver | what it is |
|---|---|
| `local` | a deterministic bag-of-words hasher. **No semantic meaning.** For tests and development. |
| `openai` | any OpenAI-compatible `POST {base}/embeddings`, through `HttpClientPort`. |

```php
// A service that declares  "requires": ["embedding.generation"]   (+ "http.client" for openai)
$vector  = $embeddings->embed('the text');                 // Vector
$vectors = $embeddings->embedMany(['a', 'b', 'c']);        // list<Vector>, input order, ONE provider call for the misses
$embeddings->dimensions();  $embeddings->model();          // what to size / key a collection on
```

It deliberately does **nothing else**: no prompts, no tools, no steps, no provider
routing. `EmbeddingPort` is `embed`, `embedMany`, `dimensions`, `model`. Anything richer
belongs to a toolkit built on top of it.

---

## No live provider was called

Say it plainly: **this plugin has never made a request to a real embedding provider.**
The `openai` adapter is verified against a *recording* `HttpClientPort` — the exact
request it sends is asserted, and canned answers are read back. The request and response
shapes (`{"model","input":[…],"encoding_format","dimensions"?}` out;
`{"data":[{"index","embedding"}]}` back) are OpenAI's published API **as the author
recalls it**, not checked against a live endpoint or against any compatible server's
documentation. A provider that deviates produces an `EmbeddingException` on the first
call — every response is validated for count, order and length — never a wrong vector.

## Configuration

| key | default | |
|---|---|---|
| `EMBEDDINGS_DRIVER` | **none — required** | `local` or `openai` |
| `EMBEDDINGS_MODEL` | per driver (`local-hash-v1` / `text-embedding-3-small`) | pin a name, not an alias |
| `EMBEDDINGS_DIMENSIONS` | per driver (256 / 1536) | sent to the provider as `dimensions` **only if you set it** |
| `EMBEDDINGS_API_KEY` | **none** | the credential; only the `openai` driver reads it |
| `EMBEDDINGS_BASE_URL` | `https://api.openai.com/v1` | any OpenAI-compatible server (Azure, vLLM, Ollama's `/v1`) |
| `EMBEDDINGS_BATCH_SIZE` | `96` | inputs per request (clamped 1..2048) |
| `EMBEDDINGS_TIMEOUT` | `30` | seconds |
| `EMBEDDINGS_MAX_INPUT_CHARS` | `24000` | longer text is refused before any request |
| `EMBEDDINGS_CACHE` | `true` | |
| `EMBEDDINGS_CACHE_TTL` | `2592000` | seconds (30 days); `0` = no expiry |

**`EMBEDDINGS_DRIVER` has no default, on purpose.** `local` as a default would let a
production system run on vectors that index and search without a single error while
carrying no meaning. `openai` as a default would make enabling the plugin a paid-API
commitment. `hkm plugins enable` writes the key empty, so the boot fails until somebody
chooses.

`EMBEDDINGS_DIMENSIONS` is capped at **2000**, the most a `VectorStorePort` collection
(pgvector HNSW) takes. A 3072-dimension model needs to be asked for fewer
(`text-embedding-3-large` accepts a `dimensions` parameter — recalled, not checked).

## The cache

Key: `sha256("emb" NUL "v1" NUL model NUL dimensions NUL text)`, prefixed `emb:`.
Pinned by a test whose expected hash came from Python.

- **Content-addressed.** An unchanged text hits wherever it moved to; a changed one
  misses with nothing needing to notice it changed.
- **Model and dimension are in the key.** Switching `EMBEDDINGS_MODEL` starts cold
  instead of serving the old model's vectors to the new model's index.
- **The key is a hash, not the text** — keys show up in `KEYS *`, monitoring and
  slow-logs. **The cached value is not harmless either**: an embedding approximates its
  text and can be inverted. Treat the cache as exactly as sensitive as the text.
- **Lossless.** Stored as base64 of packed float64, so a hit is bit-identical to what the
  provider returned.
- **Batches.** `embedMany` looks every text up, sends the *distinct misses* to the
  provider in one call (chunked by `EMBEDDINGS_BATCH_SIZE`), stores each, and returns
  results in input order. The same text twice in a batch is embedded once.
- **An optimisation, never a dependency.** A cache that throws is a miss; a store that
  throws is ignored. Both are counted (`CachingEmbedder::stats()['cacheErrors']`), because
  a dead cache looks exactly like a cold one. An entry that does not decode to exactly
  `dimensions()` floats is a miss and is overwritten.
- It uses the application's `CachePort`. With none bound, or `EMBEDDINGS_CACHE=false`,
  every call goes to the provider.

## The credential

The API key is never a string property. `EmbeddingsConfig` holds a closure that calls
`env('EMBEDDINGS_API_KEY')` when the Authorization header is built, so `var_export()`,
`json_encode()`, `print_r()`, `var_dump()` and `serialize()` of the config cannot show it
(**the first version held it in a private property and `var_export` printed it** — found
by a test, fixed). It is not in `model()`, not in any exception message, and not in the
`embeddings:probe` output. Provider error **text** is not copied into exceptions either:
OpenAI's 401 quotes a masked copy of the key it was sent (recalled), and exception
messages end up in logs; only the HTTP status and a short machine `code` are kept.

## One attempt

`retry => 0`. Whether to retry a paid call, and how often, belongs to the caller — a queue
job has its own policy. The cache makes a retried batch cheap: what already succeeded is
stored.

## The local embedder, exactly

Feature hashing: lower-case → maximal runs of letters/digits → for each token,
`sha256`, index = big-endian uint32 of bytes 0..3 mod dimensions, sign from the low bit of
byte 4 → sum → unit length. Texts sharing words get positive cosine; disjoint texts are
near-orthogonal; *car* and *automobile* are as unrelated as *car* and *banana*. If there
are no tokens, or they cancel to exactly zero, the whole text is hashed as one token — a
zero vector never leaves it. Pinned by a known-answer test computed by an independent
Python implementation. Any change to the algorithm changes every vector, so it must be a
new model name, never an edit.

## CLI

`hkm cli embeddings:probe "some text"` — shows the configuration (key: *set* / *not
set*, never the value) and, for `local`, embeds twice and reports the cache. For `openai`
it prints the request **that would be sent**, credential masked, and stops: **`--live` is
required to make the billed call**.

## Wiring

On demand: `"requires": ["embedding.generation"]`, plus `"http.client"` on the route/job
that uses the `openai` driver (without it the failure names exactly that line).
`EmbeddingPort` is a public binding; the config object — which can fetch the key — is an
internal one.

**None of the kernel's three traps applies here**, each for a stated reason (see
`Provider`): no job is registered (and a future re-embedding job must be registered with
`$worker->job('name', Class::class)` — `jobs[]` does not do it, and an unregistered name
is *skipped*, which *acks* the message — and bound with `bind()`), and no EventBus
listener exists (one would need `bind()` too).

## Tests

```
vendor/bin/phpunit
```

84 tests; no network, no database. The `openai` adapter runs against a recording
`HttpClientPort`.

## Verified / NOT verified

**Verified:** the local embedder against independent Python values; the exact request the
`openai` adapter sends (URL, headers, JSON body, timeouts, retry, response-size cap);
batching and order (including a provider answering out of order); every malformed-response
case refused; the key absent from exceptions, dumps, the probe's output and `model()`; the
cache — hit/miss/dedupe/ordering/TTL/key format/corrupt entries/failing cache; the
provider's bindings in a real kernel container (public contract, internal config); the
probe command executed.

**NOT verified:**
- **Any live provider.** No request left this machine. The request/response shapes, the
  `dimensions` and `encoding_format` parameters, the 2048-input and 8191-token limits
  mentioned in comments, and the claim that an empty `input` is rejected are all recalled
  from documentation, not tested.
- Any server other than the recording client — Ollama, vLLM, Azure's differing URL scheme
  and `api-key` header (Azure is **not** supported by this adapter as written).
- A real `CachePort` adapter (Redis, files): only `FakeCache` was used. Whether a given
  adapter round-trips a base64 string unchanged is its own contract.
- **Concurrent callers.** Two requests embedding the same text at once both miss and both
  pay; there is no single-flight lock.
- Retry behaviour, rate limiting and token counting: none exist here. The character cap
  (24000) is a heuristic, not a token count.
