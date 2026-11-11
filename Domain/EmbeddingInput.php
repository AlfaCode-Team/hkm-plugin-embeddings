<?php

declare(strict_types=1);

namespace Plugins\Embeddings\Domain;

/**
 * The rules every embedder applies to its input, in ONE place.
 *
 * The caching decorator applies the same rule as the adapters it wraps: a
 * decorator that accepted what its inner adapter refuses would turn a cache HIT
 * into a different outcome from a cache MISS, and the difference would depend on
 * what happened to be cached.
 */
final class EmbeddingInput
{
    private function __construct() {}

    /**
     * Shared input rules: a list, no blank text, nothing over the length cap.
     * Static and public so the HTTP adapter and the caching decorator apply the
     * SAME rule — a decorator that accepted what its inner adapter refuses would
     * turn a cache hit into a different outcome from a cache miss.
     *
     * @param array<array-key, mixed> $texts
     * @throws \InvalidArgumentException
     */
    public static function assertTexts(array $texts, int $maxChars): void
    {
        if (!array_is_list($texts)) {
            throw new \InvalidArgumentException('embedMany() takes a list of strings; the position of a text is how its vector is found again.');
        }

        foreach ($texts as $i => $text) {
            if (!\is_string($text)) {
                throw new \InvalidArgumentException(\sprintf('Text %d is %s, not a string.', $i, get_debug_type($text)));
            }

            if (trim($text) === '') {
                throw new \InvalidArgumentException(\sprintf(
                    'Text %d is blank. Embedding nothing yields a meaningless vector (or a refusal, or a charge), so it is refused here.',
                    $i,
                ));
            }

            if (mb_strlen($text) > $maxChars) {
                throw new \InvalidArgumentException(\sprintf(
                    'Text %d is %d characters; the limit is %d (EMBEDDINGS_MAX_INPUT_CHARS). Split it into chunks — a provider would reject it, and one vector for a whole document is rarely what is wanted.',
                    $i,
                    mb_strlen($text),
                    $maxChars,
                ));
            }
        }
    }
}
