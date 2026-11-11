<?php

declare(strict_types=1);

namespace Plugins\Embeddings\Infrastructure;

use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\CachePort;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\EmbeddingPort;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\HttpClientPort;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\Vector\EmbeddingException;
use Plugins\Embeddings\Application\CachingEmbedder;
use Plugins\Embeddings\Infrastructure\Config\EmbeddingsConfig;
use Plugins\Embeddings\Infrastructure\Gateways\LocalHashEmbedder;
use Plugins\Embeddings\Infrastructure\Gateways\OpenAiCompatibleEmbedder;

/**
 * Builds the embedder a configuration describes. One place, so the Provider and
 * the CLI command cannot disagree about what "the openai driver" means.
 */
final class EmbedderFactory
{
    private function __construct() {}

    /** The adapter itself, without the cache. */
    public static function inner(EmbeddingsConfig $config, ?HttpClientPort $http): EmbeddingPort
    {
        if ($config->driver === EmbeddingsConfig::DRIVER_LOCAL) {
            return new LocalHashEmbedder($config->dimensions, $config->maxInputChars);
        }

        if ($http === null) {
            throw new EmbeddingException(
                'The openai driver needs an HTTP client, and none is available in this scope. Declare '
                . '"requires": ["embedding.generation", "http.client"] on the route (or job) that uses it.',
                'config',
            );
        }

        return new OpenAiCompatibleEmbedder($http, $config);
    }

    /** The adapter, wrapped in the content-hash cache when one is configured AND available. */
    public static function build(EmbeddingsConfig $config, ?HttpClientPort $http, ?CachePort $cache): EmbeddingPort
    {
        $inner = self::inner($config, $http);

        if (!$config->cache || $cache === null) {
            return $inner;
        }

        return new CachingEmbedder($inner, $cache, $config->cacheTtl, $config->maxInputChars);
    }
}
