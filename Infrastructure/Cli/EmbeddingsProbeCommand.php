<?php

declare(strict_types=1);

namespace Plugins\Embeddings\Infrastructure\Cli;

use AlfacodeTeam\PhpIoCli\AbstractCommand;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\EmbeddingPort;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\Vector\EmbeddingException;
use Plugins\Embeddings\Application\CachingEmbedder;
use Plugins\Embeddings\Infrastructure\Config\EmbeddingsConfig;
use Plugins\Embeddings\Infrastructure\Gateways\OpenAiCompatibleEmbedder;

/**
 * `embeddings:probe` — show what is configured, and run one text through it.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * IT NEVER SPENDS MONEY UNLESS TOLD TO
 * ────────────────────────────────────────────────────────────────────────────
 * With the `local` driver it embeds the text twice and reports the cache. With
 * `openai` it prints the request that WOULD be sent — credential masked — and
 * stops; `--live` is required to make the call, because that is a billed request
 * and a probe run in a deploy script should not be one by accident.
 *
 * The API key is reported as "set" or "not set", never shown.
 */
final class EmbeddingsProbeCommand extends AbstractCommand
{
    /**
     * @param \Closure(): EmbeddingsConfig                              $config
     * @param \Closure(EmbeddingsConfig): array{0: EmbeddingPort, 1: EmbeddingPort} $build [inner adapter, cached embedder]
     */
    public function __construct(
        private readonly \Closure $config,
        private readonly \Closure $build,
    ) {
        parent::__construct();
    }

    protected function configure(): void
    {
        $this->name        = 'embeddings:probe';
        $this->description = 'Show the embedding configuration and embed one text (the openai driver needs --live).';

        $this->addArgument('text', 'The text to embed.', false, 'hello world');
        $this->addOption('live', '', 'Make the real, billed provider call (openai driver only).');
    }

    protected function handle(): int
    {
        try {
            $config = ($this->config)();
            [$inner, $embedder] = ($this->build)($config);
        } catch (EmbeddingException $e) {
            $this->error($e->getMessage());

            return self::FAILURE;
        }

        $text = (string) $this->argument('text', 'hello world');

        $this->section('Configuration');
        $this->info('driver:     ' . $config->driver);
        $this->info('model:      ' . $embedder->model());
        $this->info('dimensions: ' . $embedder->dimensions());
        $this->info('cache:      ' . ($embedder instanceof CachingEmbedder ? 'on (content-hash keyed)' : 'off'));
        $this->info('api key:    ' . ($config->hasApiKey() ? 'set' : 'not set'));

        if ($config->driver === EmbeddingsConfig::DRIVER_OPENAI && $inner instanceof OpenAiCompatibleEmbedder && !$this->option('live')) {
            $this->section('Request that would be sent');
            $this->info(json_encode($inner->describeRequest([$text]), JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE) ?: '');
            $this->muted('No request was made. Pass --live to call the provider (this is billed).');

            return self::SUCCESS;
        }

        try {
            $first  = $embedder->embed($text);
            $second = $embedder->embed($text);
        } catch (\InvalidArgumentException | EmbeddingException $e) {
            $this->error($e->getMessage());

            return self::FAILURE;
        }

        $this->section('Result');
        $this->info(\sprintf('%d dimensions, first values: %s', $first->dimensions(), implode(', ', array_map(
            static fn (float $x): string => rtrim(rtrim(\sprintf('%.5f', $x), '0'), '.'),
            \array_slice($first->toArray(), 0, 5),
        ))));
        $this->info('second call returned an identical vector: ' . ($first->toArray() === $second->toArray() ? 'yes' : 'NO'));

        if ($embedder instanceof CachingEmbedder) {
            $s = $embedder->stats();
            $this->info(\sprintf('cache: %d hit(s), %d miss(es), %d provider call(s)', $s['hits'], $s['misses'], $s['providerCalls']));
        }

        if ($config->driver === EmbeddingsConfig::DRIVER_LOCAL) {
            $this->warning('The local driver has NO semantic meaning. Do not use it outside tests and development.');
        }

        return self::SUCCESS;
    }
}
