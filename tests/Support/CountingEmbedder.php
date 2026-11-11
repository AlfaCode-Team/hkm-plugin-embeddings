<?php

declare(strict_types=1);

namespace Tests\Plugins\Embeddings\Support;

use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\EmbeddingPort;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\Vector\Vector;

/**
 * An EmbeddingPort that counts what it is asked — the "provider" behind the cache.
 *
 * Its vectors are a function of the text and the model, so a changed model gives
 * changed vectors, which is what lets a test see a wrong cache hit as a wrong
 * VALUE and not only as a wrong call count.
 */
class CountingEmbedder implements EmbeddingPort
{
    /** @var list<list<string>> every embedMany() call, with its inputs */
    public array $calls = [];

    public function __construct(private readonly string $model = 'test-model', private readonly int $dims = 4) {}

    public function embed(string $text): Vector
    {
        return $this->embedMany([$text])[0];
    }

    public function embedMany(array $texts): array
    {
        $this->calls[] = $texts;

        return array_map(function (string $t): Vector {
            $h = hash('sha256', $this->model . "\0" . $t, true);
            $v = [];

            for ($i = 0; $i < $this->dims; $i++) {
                $v[] = (\ord($h[$i]) + 1) / 257.0 + $i * 1e-9; // non-zero, and not representable in float32
            }

            return Vector::of($v);
        }, $texts);
    }

    public function dimensions(): int
    {
        return $this->dims;
    }

    public function model(): string
    {
        return $this->model;
    }

    public function inputCount(): int
    {
        return array_sum(array_map('count', $this->calls));
    }
}
