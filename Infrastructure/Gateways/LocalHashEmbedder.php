<?php

declare(strict_types=1);

namespace Plugins\Embeddings\Infrastructure\Gateways;

use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\EmbeddingPort;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\Vector\Vector;
use Plugins\Embeddings\Domain\EmbeddingInput;

/**
 * A deterministic embedder with NO semantic meaning.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * WHAT IT IS, AND WHAT IT IS NOT
 * ════════════════════════════════════════════════════════════════════════════
 * Feature hashing of a bag of words: each lower-cased word is hashed to one
 * coordinate and a sign, the counts are summed, and the result is scaled to unit
 * length. So two texts SHARING WORDS get vectors with positive cosine similarity,
 * and texts sharing none are (near) orthogonal. That is the whole of it. "Car"
 * and "automobile" are as unrelated as "car" and "banana".
 *
 * It exists so the pipeline around an embedder — the cache, the batching, the
 * vector store, the dimension checks — can be run and tested without a network
 * and without paying anybody. It must not run in production, and nothing inside
 * a store can tell that it is: the vectors index and search without complaint.
 * That is why EMBEDDINGS_DRIVER has no default.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * THE ALGORITHM (v1) — pinned by a known-answer test computed in Python
 * ════════════════════════════════════════════════════════════════════════════
 *   tokens  = every maximal run of Unicode letters/digits in mb_strtolower(text)
 *   for each token:  h = sha256(token)
 *                    index = big-endian uint32 of h[0..3]  mod dimensions
 *                    sign  = +1 if the low bit of h[4] is 0, else -1
 *                    v[index] += sign
 *   v = v / |v|
 * If there are no tokens, or they cancel to exactly zero, the whole text is
 * treated as ONE token (sha256 of the raw text). A zero vector must never leave
 * here: a cosine collection refuses it, and a store that did not would sort it
 * after everything for ever.
 *
 * Changing ANY of this changes every vector, so it changes {@see self::model()},
 * which is part of the cache key — never edit the algorithm in place; add v2.
 */
final class LocalHashEmbedder implements EmbeddingPort
{
    public const MODEL = 'local-hash-v1';

    public function __construct(
        private readonly int $dimensions = 256,
        private readonly int $maxInputChars = 24000,
    ) {
        if ($dimensions < 8 || $dimensions > Vector::MAX_DIMENSIONS) {
            throw new \InvalidArgumentException(\sprintf(
                'The local embedder uses between 8 and %d dimensions; %d was given. Below 8 almost '
                . 'every word collides with another.',
                Vector::MAX_DIMENSIONS,
                $dimensions,
            ));
        }
    }

    public function embed(string $text): Vector
    {
        return $this->embedMany([$text])[0];
    }

    public function embedMany(array $texts): array
    {
        EmbeddingInput::assertTexts($texts, $this->maxInputChars);

        return array_map($this->one(...), $texts);
    }

    public function dimensions(): int
    {
        return $this->dimensions;
    }

    public function model(): string
    {
        return self::MODEL;
    }

    private function one(string $text): Vector
    {
        preg_match_all('/[\p{L}\p{N}]+/u', mb_strtolower($text), $m);

        $v = $this->accumulate($m[0]);

        if ($v === null) {
            // No tokens, or they cancelled exactly: the text as a single token.
            $v = $this->accumulate([$text]) ?? throw new \LogicException('unreachable: one token cannot cancel');
        }

        return Vector::of($v);
    }

    /**
     * @param list<string> $tokens
     * @return list<float>|null null when the sum is the zero vector
     */
    private function accumulate(array $tokens): ?array
    {
        $v = array_fill(0, $this->dimensions, 0.0);

        foreach ($tokens as $token) {
            $h             = hash('sha256', $token, true);
            $index         = unpack('N', substr($h, 0, 4))[1] % $this->dimensions;
            $v[$index]    += (\ord($h[4]) & 1) === 0 ? 1.0 : -1.0;
        }

        $norm = 0.0;

        foreach ($v as $x) {
            $norm += $x * $x;
        }

        if ($norm === 0.0) {
            return null;
        }

        $norm = sqrt($norm);

        return array_map(static fn (float $x): float => $x / $norm, $v);
    }
}
