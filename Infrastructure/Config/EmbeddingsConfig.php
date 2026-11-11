<?php

declare(strict_types=1);

namespace Plugins\Embeddings\Infrastructure\Config;

use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\Vector\EmbeddingException;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\Vector\Vector;

/**
 * Every knob, normalised in the private constructor.
 *
 * `env()` and not `getenv()`: `$_ENV` is the source of truth and `putenv()` is
 * never called, so `getenv()` returns null for a value that is present.
 *
 * THE API KEY IS NOT A PROPERTY. This class holds a closure that FETCHES it, and
 * for the production path that closure captures nothing — it calls `env()` when
 * asked. Measured the hard way: the first version held the key in a private
 * property, and `var_export($config)` printed it (var_export, `serialize()` and an
 * `(array)` cast all read private properties and ignore `__debugInfo()`). A closure
 * is invisible to `var_export` and refuses `serialize()`, and a closure that
 * captures nothing has nothing for a debugger to show.
 */
final class EmbeddingsConfig
{
    public const DRIVER_LOCAL  = 'local';
    public const DRIVER_OPENAI = 'openai';

    public const LOCAL_MODEL            = 'local-hash-v1';
    public const LOCAL_DIMENSIONS       = 256;
    public const OPENAI_DEFAULT_MODEL   = 'text-embedding-3-small';
    public const OPENAI_DEFAULT_DIMS    = 1536;

    public readonly string $driver;
    public readonly string $model;
    public readonly int $dimensions;

    /** True only when EMBEDDINGS_DIMENSIONS was set: the `dimensions` request parameter is then sent. */
    public readonly bool $dimensionsExplicit;

    public readonly string $baseUrl;
    public readonly int $batchSize;
    public readonly int $timeout;
    public readonly int $maxInputChars;
    public readonly bool $cache;

    /** Seconds; null = no expiry. */
    public readonly ?int $cacheTtl;

    /** @var \Closure(): string returns the key; never stored as a string here */
    private readonly \Closure $apiKeySource;

    private function __construct(
        string $driver,
        ?string $model,
        ?int $dimensions,
        \Closure $apiKeySource,
        string $baseUrl,
        int $batchSize,
        int $timeout,
        int $maxInputChars,
        bool $cache,
        int $cacheTtl,
    ) {
        if (!\in_array($driver, [self::DRIVER_LOCAL, self::DRIVER_OPENAI], true)) {
            throw new EmbeddingException(
                'EMBEDDINGS_DRIVER must be "local" or "openai"; got ' . ($driver === '' ? 'an empty value' : '"' . substr($driver, 0, 32) . '"')
                . '. It has no default on purpose — see module.json.',
                'config',
            );
        }

        $this->driver = $driver;
        $this->model  = $model !== null && $model !== ''
            ? $model
            : ($driver === self::DRIVER_LOCAL ? self::LOCAL_MODEL : self::OPENAI_DEFAULT_MODEL);

        $default                  = $driver === self::DRIVER_LOCAL ? self::LOCAL_DIMENSIONS : self::OPENAI_DEFAULT_DIMS;
        $this->dimensionsExplicit = $dimensions !== null && $dimensions > 0;
        $this->dimensions         = $this->dimensionsExplicit ? min($dimensions, Vector::MAX_DIMENSIONS) : $default;

        $this->apiKeySource  = $apiKeySource;
        $this->baseUrl       = rtrim($baseUrl, '/');
        // 1..2048: the second figure is OpenAI's documented per-request input limit (recalled, not
        // checked here); a configured value beyond what a provider takes would fail on the first batch.
        $this->batchSize     = max(1, min(2048, $batchSize));
        $this->timeout       = max(1, min(300, $timeout));
        $this->maxInputChars = max(1, $maxInputChars);
        $this->cache         = $cache;
        $this->cacheTtl      = $cacheTtl > 0 ? $cacheTtl : null;
    }

    public static function fromEnv(): self
    {
        $dims = env('EMBEDDINGS_DIMENSIONS');

        return new self(
            driver:        strtolower(trim((string) env('EMBEDDINGS_DRIVER', ''))),
            model:         ($m = env('EMBEDDINGS_MODEL')) === null ? null : (string) $m,
            dimensions:    $dims === null || $dims === '' ? null : (int) $dims,
            // Captures nothing: the key stays in the environment and is read when it is needed.
            apiKeySource:  static fn (): string => (string) env('EMBEDDINGS_API_KEY', ''),
            baseUrl:       (string) env('EMBEDDINGS_BASE_URL', 'https://api.openai.com/v1'),
            batchSize:     (int) env('EMBEDDINGS_BATCH_SIZE', 96),
            timeout:       (int) env('EMBEDDINGS_TIMEOUT', 30),
            maxInputChars: (int) env('EMBEDDINGS_MAX_INPUT_CHARS', 24000),
            // filter_var, not (bool): (bool)"false" is TRUE in PHP, and a cache that cannot be
            // switched off by writing `false` in .env is a defect that only shows when it is needed.
            cache:         filter_var(env('EMBEDDINGS_CACHE', true), FILTER_VALIDATE_BOOLEAN, FILTER_NULL_ON_FAILURE) ?? true,
            cacheTtl:      (int) env('EMBEDDINGS_CACHE_TTL', 2592000),
        );
    }

    public static function of(
        string $driver = self::DRIVER_LOCAL,
        ?string $model = null,
        ?int $dimensions = null,
        string $apiKey = '',
        string $baseUrl = 'https://api.openai.com/v1',
        int $batchSize = 96,
        int $timeout = 30,
        int $maxInputChars = 24000,
        bool $cache = true,
        int $cacheTtl = 2592000,
    ): self {
        return new self($driver, $model, $dimensions, static fn (): string => $apiKey, $baseUrl, $batchSize, $timeout, $maxInputChars, $cache, $cacheTtl);
    }

    public function hasApiKey(): bool
    {
        return ($this->apiKeySource)() !== '';
    }

    /**
     * The key, for the one caller that must put it in an Authorization header.
     * Named so that a grep for `revealApiKey` finds every place it can leave.
     */
    public function revealApiKey(): string
    {
        return ($this->apiKeySource)();
    }

    /** @return array<string, mixed> */
    public function __debugInfo(): array
    {
        return ['driver' => $this->driver, 'model' => $this->model, 'dimensions' => $this->dimensions, 'apiKey' => $this->hasApiKey() ? '(set)' : '(none)'];
    }
}
