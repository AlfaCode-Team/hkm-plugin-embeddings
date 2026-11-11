<?php

declare(strict_types=1);

namespace Plugins\Embeddings;

use AlfacodeTeam\PhpServicePlatform\Kernel\Container\ModuleContainer;
use AlfacodeTeam\PhpServicePlatform\Kernel\Contracts\ModuleContract;
use AlfacodeTeam\PhpServicePlatform\Kernel\Events\EventBus;
use AlfacodeTeam\PhpServicePlatform\Kernel\Pipelines\Cli\CliPipeline;
use AlfacodeTeam\PhpServicePlatform\Kernel\Pipelines\Http\HttpPipeline;
use AlfacodeTeam\PhpServicePlatform\Kernel\Pipelines\Worker\WorkerPipeline;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\CachePort;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\EmbeddingPort;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\HttpClientPort;
use Plugins\Embeddings\Infrastructure\Cli\EmbeddingsProbeCommand;
use Plugins\Embeddings\Infrastructure\Config\EmbeddingsConfig;
use Plugins\Embeddings\Infrastructure\EmbedderFactory;

/**
 * Embeddings — binds EmbeddingPort, behind a content-hash cache.
 *
 * ACTIVATION: an on-demand module.
 *
 *   { "requires": ["embedding.generation"] }                       // local driver
 *   { "requires": ["embedding.generation", "http.client"] }        // openai driver
 *
 * `http.client` is not in this module's own requires[], so a route that only
 * uses the local driver does not load an HTTP client it never calls — the same
 * optional-port pattern as the avatars plugin.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * WHICH OF THE KERNEL'S THREE TRAPS APPLY HERE
 * ────────────────────────────────────────────────────────────────────────────
 * None, each for a stated reason:
 *
 *  • No job is registered. A batch RE-embedding job belongs to whoever owns the
 *    source text; when one is written it must be registered with
 *    `$worker->job('name', Class::class)` in boot() — `jobs[]` in module.json
 *    does NOT register a handler, an unregistered name resolves to nothing, and
 *    WorkerLoop treats that as "skipped", which ACKS the message and discards it.
 *    The handler would also be bound with `bind()`, because WorkerLoop resolves
 *    it from outside this module's scope.
 *  • No EventBus listener (`listens` is empty). Were one added it would need
 *    `bind()` too: EventBus resolves listeners from the kernel scope.
 *  • EmbeddingPort itself is bound with `bind()` — it is the public contract.
 *
 * The config (and with it the API key) is an INTERNAL binding: nothing outside
 * this module can resolve the object that holds the credential.
 */
final class Provider implements ModuleContract
{
    public function solves(): string
    {
        return 'embedding.generation';
    }

    /** @return list<string> */
    public function requires(): array
    {
        return [];
    }

    /** @return list<class-string> */
    public function exposes(): array
    {
        return [EmbeddingPort::class];
    }

    public function register(ModuleContainer $container): void
    {
        if ($container->has(EmbeddingPort::class)) {
            return; // a project already provided its own EmbeddingPort
        }

        $container->bindInternal(
            EmbeddingsConfig::class,
            static fn (): EmbeddingsConfig => EmbeddingsConfig::fromEnv(),
        );

        $container->bind(
            EmbeddingPort::class,
            static function (ModuleContainer $c): EmbeddingPort {
                $config = $c->make(EmbeddingsConfig::class);

                return EmbedderFactory::build(
                    $config,
                    // `has()` and not a try/catch: a missing HTTP client is the expected state for
                    // the local driver, and EmbedderFactory turns it into a message that names the
                    // line to add when the openai driver actually needs it.
                    $c->has(HttpClientPort::class) ? $c->make(HttpClientPort::class) : null,
                    $c->has(CachePort::class) ? $c->make(CachePort::class) : null,
                );
            },
        );
    }

    public function boot(HttpPipeline $http, CliPipeline $cli, WorkerPipeline $worker, EventBus $events): void
    {
        // Deferred, registered as an INSTANCE: a command registered by class-string is built from
        // the CoreContainer, where EmbeddingPort (bound per request) does not exist.
        $cli->defer(static function (CliPipeline $cli): void {
            $core = $cli->container();

            $cli->command(new EmbeddingsProbeCommand(
                static fn (): EmbeddingsConfig => EmbeddingsConfig::fromEnv(),
                static fn (EmbeddingsConfig $config): array => [
                    EmbedderFactory::inner($config, $core->has(HttpClientPort::class) ? $core->make(HttpClientPort::class) : null),
                    EmbedderFactory::build(
                        $config,
                        $core->has(HttpClientPort::class) ? $core->make(HttpClientPort::class) : null,
                        $core->has(CachePort::class) ? $core->make(CachePort::class) : null,
                    ),
                ],
            ));
        });
    }
}
