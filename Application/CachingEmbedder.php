<?php

declare(strict_types=1);

namespace Plugins\Embeddings\Application;

use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\CachePort;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\EmbeddingPort;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\Vector\EmbeddingException;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\Vector\Vector;
use Plugins\Embeddings\Domain\EmbeddingInput;

/**
 * An EmbeddingPort that never embeds the same text twice.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * THE KEY
 * ════════════════════════════════════════════════════════════════════════════
 *   sha256( "emb" NUL "v1" NUL model NUL dimensions NUL text )
 *
 * Content-addressed, so an UNCHANGED text hits wherever it moved to and a changed
 * one misses without anything having to notice it changed. The model and the
 * dimension are in the key because vectors from two models are not
 * interchangeable even at one length: switching EMBEDDINGS_MODEL starts a cold
 * cache instead of serving the old model's vectors to the new model's index.
 *
 * The key is a hash, not the text: a cache key is visible in `KEYS *`, in
 * monitoring and in slow-logs, and the text is often the private thing.
 *
 * THE CACHED VALUE IS NOT HARMLESS EITHER. An embedding approximates its text
 * and can be inverted to a recognisable reconstruction; treat the cache as being
 * exactly as sensitive as the text it was made from.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * ONE PROVIDER CALL PER BATCH OF MISSES — AND DUPLICATES COST NOTHING
 * ════════════════════════════════════════════════════════════════════════════
 * `embedMany()` looks every text up, collects the DISTINCT misses in first-seen
 * order, sends them to the inner embedder in one call, stores each, and puts the
 * results back in input order. The same text twice in one batch is embedded once
 * (and counted as one miss and one hit).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * THE CACHE IS AN OPTIMISATION, NEVER A DEPENDENCY
 * ════════════════════════════════════════════════════════════════════════════
 * A cache that throws (Redis down) is a miss; a store that throws is ignored.
 * Both are counted in {@see self::stats()}, because a cache that has been failing
 * silently for a week looks exactly like a cache with a bad hit rate. A cached
 * value that does not decode to exactly `dimensions()` floats is also a miss —
 * it is from another configuration or corrupt, and returning it would put a
 * wrong-length vector into a collection.
 *
 * The value is stored as base64 of packed float64 (lossless: a hit returns the
 * bit-identical vector the provider returned, which float32 packing or JSON
 * would not guarantee), so it survives any CachePort adapter that serialises
 * strings.
 */
final class CachingEmbedder implements EmbeddingPort
{
    private const KEY_VERSION = 'v1';

    private int $hits = 0;

    private int $misses = 0;

    private int $providerCalls = 0;

    private int $providerInputs = 0;

    private int $cacheErrors = 0;

    public function __construct(
        private readonly EmbeddingPort $inner,
        private readonly CachePort $cache,
        private readonly ?int $ttl = null,
        private readonly int $maxInputChars = 24000,
        private readonly string $namespace = 'emb',
    ) {}

    public function embed(string $text): Vector
    {
        return $this->embedMany([$text])[0];
    }

    public function embedMany(array $texts): array
    {
        // The SAME rule as the adapters: a hit must not succeed where a miss would have refused.
        EmbeddingInput::assertTexts($texts, $this->maxInputChars);

        if ($texts === []) {
            return [];
        }

        $dimensions = $this->inner->dimensions();
        $keys       = array_map($this->keyFor(...), $texts);

        /** @var array<string, Vector> $found key => vector */
        $found = [];
        /** @var array<string, string> $missing key => text, first-seen order, distinct */
        $missing = [];

        foreach ($texts as $i => $text) {
            $key = $keys[$i];

            // `isset($missing[$key])` is redundant for correctness ($missing is keyed by hash, so a
            // repeat collapses by itself); it saves a second cache round trip for the same key.
            // Pinned by counting lookups — no result-level test can see it.
            if (isset($found[$key]) || isset($missing[$key])) {
                continue;
            }

            $cached = $this->lookup($key, $dimensions);

            if ($cached !== null) {
                $found[$key] = $cached;
            } else {
                $missing[$key] = $text;
            }
        }

        if ($missing !== []) {
            $fresh = $this->inner->embedMany(array_values($missing));

            if (\count($fresh) !== \count($missing)) {
                throw new EmbeddingException(
                    \sprintf('The embedder returned %d vectors for %d inputs.', \count($fresh), \count($missing)),
                    'embed',
                );
            }

            $this->providerCalls++;
            $this->providerInputs += \count($missing);

            foreach (array_keys($missing) as $n => $key) {
                $found[$key] = $fresh[$n];
                $this->store($key, $fresh[$n]);
            }
        }

        // Counted per INPUT, so hits + misses == the number of texts asked for; a duplicate of a
        // text that missed is a hit on the vector the first copy just paid for.
        $firstMiss = $missing;

        $out = [];

        foreach ($keys as $key) {
            if (isset($firstMiss[$key])) {
                $this->misses++;
                unset($firstMiss[$key]);
            } else {
                $this->hits++;
            }

            $out[] = $found[$key];
        }

        return $out;
    }

    public function dimensions(): int
    {
        return $this->inner->dimensions();
    }

    public function model(): string
    {
        return $this->inner->model();
    }

    /** @return array{hits: int, misses: int, providerCalls: int, providerInputs: int, cacheErrors: int} */
    public function stats(): array
    {
        return [
            'hits'           => $this->hits,
            'misses'         => $this->misses,
            'providerCalls'  => $this->providerCalls,
            'providerInputs' => $this->providerInputs,
            'cacheErrors'    => $this->cacheErrors,
        ];
    }

    /** The cache key for a text under THIS embedder's model and dimension. Public so tests pin the format. */
    public function keyFor(string $text): string
    {
        return $this->namespace . ':' . hash('sha256', implode("\0", [
            'emb',
            self::KEY_VERSION,
            $this->inner->model(),
            (string) $this->inner->dimensions(),
            $text,
        ]));
    }

    private function lookup(string $key, int $dimensions): ?Vector
    {
        try {
            $raw = $this->cache->get($key);
        } catch (\Throwable) {
            $this->cacheErrors++;

            return null;
        }

        if (!\is_string($raw)) {
            return null;
        }

        $bytes = base64_decode($raw, true);

        if ($bytes === false || \strlen($bytes) !== $dimensions * 8) {
            return null;
        }

        $values = unpack('d*', $bytes);

        if ($values === false) {
            return null;
        }

        try {
            return Vector::of(array_values($values));
        } catch (\InvalidArgumentException) {
            return null;
        }
    }

    private function store(string $key, Vector $vector): void
    {
        try {
            $this->cache->set($key, base64_encode(pack('d*', ...$vector->toArray())), $this->ttl);
        } catch (\Throwable) {
            $this->cacheErrors++;
        }
    }
}
