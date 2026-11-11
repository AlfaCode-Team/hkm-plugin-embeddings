<?php

declare(strict_types=1);

namespace Tests\Plugins\Embeddings\Unit;

use AlfacodeTeam\Ground\Fakes\FakeCache;
use AlfacodeTeam\PhpIoCli\AbstractCommand;
use AlfacodeTeam\PhpIoCli\BufferIO;
use AlfacodeTeam\PhpServicePlatform\Kernel\Container\CoreContainer;
use AlfacodeTeam\PhpServicePlatform\Kernel\Container\ModuleContainer;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\CachePort;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\EmbeddingPort;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\HttpClientPort;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\Vector\EmbeddingException;
use PHPUnit\Framework\TestCase;
use Plugins\Embeddings\Application\CachingEmbedder;
use Plugins\Embeddings\Infrastructure\Cli\EmbeddingsProbeCommand;
use Plugins\Embeddings\Infrastructure\Config\EmbeddingsConfig;
use Plugins\Embeddings\Infrastructure\EmbedderFactory;
use Plugins\Embeddings\Infrastructure\Gateways\LocalHashEmbedder;
use Plugins\Embeddings\Infrastructure\Gateways\OpenAiCompatibleEmbedder;
use Plugins\Embeddings\Provider;
use Tests\Plugins\Embeddings\Support\RecordingHttpClient;

final class FactoryProviderAndCommandTest extends TestCase
{
    private const KEY = 'sk-probe-SECRET-0000';

    // ── factory ──────────────────────────────────────────────────────────────

    public function test_local_without_a_cache_is_the_bare_adapter_and_with_one_is_wrapped(): void
    {
        $config = EmbeddingsConfig::of(driver: 'local');

        self::assertInstanceOf(LocalHashEmbedder::class, EmbedderFactory::build($config, null, null));
        self::assertInstanceOf(CachingEmbedder::class, EmbedderFactory::build($config, null, new FakeCache()));
        self::assertInstanceOf(LocalHashEmbedder::class, EmbedderFactory::build(EmbeddingsConfig::of(driver: 'local', cache: false), null, new FakeCache()), 'EMBEDDINGS_CACHE=false switches it off');
    }

    public function test_openai_without_an_http_client_names_the_line_to_add(): void
    {
        $this->expectException(EmbeddingException::class);
        $this->expectExceptionMessageMatches('/"requires": \["embedding.generation", "http.client"\]/');

        EmbedderFactory::build(EmbeddingsConfig::of(driver: 'openai', apiKey: 'k'), null, null);
    }

    public function test_openai_with_a_client_is_the_http_adapter(): void
    {
        $e = EmbedderFactory::inner(EmbeddingsConfig::of(driver: 'openai', apiKey: 'k'), new RecordingHttpClient());

        self::assertInstanceOf(OpenAiCompatibleEmbedder::class, $e);
        self::assertSame(1536, $e->dimensions());
    }

    // ── provider, in a REAL container ────────────────────────────────────────

    public function test_the_provider_binds_a_public_embedding_port_in_a_real_container(): void
    {
        $saved = $_ENV['EMBEDDINGS_DRIVER'] ?? null;
        $_ENV['EMBEDDINGS_DRIVER'] = 'local';

        try {
            $core = new CoreContainer();
            $core->instance(CachePort::class, new FakeCache());

            $module = new ModuleContainer($core);
            $module->setScope('embedding.generation');
            (new Provider())->register($module);

            // Resolved from ANOTHER scope: bind() must be public, bindInternal() would throw here.
            $port = $module->makeInScope(EmbeddingPort::class, 'some.other.module');

            self::assertInstanceOf(EmbeddingPort::class, $port);
            self::assertInstanceOf(CachingEmbedder::class, $port, 'the cache is on by default when a CachePort exists');
            self::assertSame(256, $port->embed('hello')->dimensions());

            // The config (and so the credential) is INTERNAL: not resolvable from outside.
            $this->expectException(\AlfacodeTeam\PhpServicePlatform\Kernel\Exceptions\ScopeViolationException::class);
            $module->makeInScope(EmbeddingsConfig::class, 'some.other.module');
        } finally {
            $saved === null ? $this->unsetEnv() : $_ENV['EMBEDDINGS_DRIVER'] = $saved;
        }
    }

    public function test_a_project_supplied_port_is_not_overridden(): void
    {
        $core   = new CoreContainer();
        $module = new ModuleContainer($core);
        $mine   = new LocalHashEmbedder(16);
        $module->instance(EmbeddingPort::class, $mine);

        (new Provider())->register($module);

        self::assertSame($mine, $module->make(EmbeddingPort::class));
    }

    // ── the command, executed ────────────────────────────────────────────────

    /** @return array{code: int, output: string} */
    private function probe(EmbeddingsConfig $config, ?RecordingHttpClient $http, array $tokens = []): array
    {
        $command = new EmbeddingsProbeCommand(
            static fn (): EmbeddingsConfig => $config,
            static fn (EmbeddingsConfig $c): array => [
                EmbedderFactory::inner($c, $http),
                EmbedderFactory::build($c, $http, new FakeCache()),
            ],
        );
        $io = new BufferIO();

        ob_start();
        $code   = $command->execute($tokens, $io);
        $direct = (string) ob_get_clean();

        return ['code' => $code, 'output' => (string) preg_replace('/\e\[[0-9;]*m/', '', $direct . "\n" . $io->getOutput())];
    }

    public function test_probe_with_the_local_driver_runs_twice_and_reports_the_cache(): void
    {
        $r = $this->probe(EmbeddingsConfig::of(driver: 'local'), null, ['the quick brown fox']);

        self::assertSame(AbstractCommand::SUCCESS, $r['code']);
        self::assertStringContainsString('driver:     local', $r['output']);
        self::assertStringContainsString('dimensions: 256', $r['output']);
        self::assertStringContainsString('cache:      on (content-hash keyed)', $r['output']);
        self::assertStringContainsString('256 dimensions, first values:', $r['output']);
        self::assertStringContainsString('second call returned an identical vector: yes', $r['output']);
        self::assertStringContainsString('cache: 1 hit(s), 1 miss(es), 1 provider call(s)', $r['output']);
        self::assertStringContainsString('NO semantic meaning', $r['output']);
    }

    public function test_probe_with_openai_and_no_live_flag_prints_the_masked_request_and_calls_nothing(): void
    {
        $http = new RecordingHttpClient();
        $r    = $this->probe(EmbeddingsConfig::of(driver: 'openai', apiKey: self::KEY, dimensions: 8), $http, ['hello']);

        self::assertSame(AbstractCommand::SUCCESS, $r['code']);
        self::assertSame([], $http->calls, 'no --live: not one request');
        self::assertStringContainsString('Request that would be sent', $r['output']);
        self::assertStringContainsString('https://api.openai.com/v1/embeddings', $r['output']);
        self::assertStringContainsString('Bearer ***', $r['output']);
        self::assertStringContainsString('api key:    set', $r['output']);
        self::assertStringContainsString('No request was made', $r['output']);
        self::assertStringNotContainsString(self::KEY, $r['output']);
    }

    public function test_probe_with_live_makes_the_call_and_still_never_prints_the_key(): void
    {
        $http = (new RecordingHttpClient())->queue(RecordingHttpClient::embeddingsResponse([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]));
        $r    = $this->probe(EmbeddingsConfig::of(driver: 'openai', apiKey: self::KEY, dimensions: 8), $http, ['hello', '--live']);

        self::assertSame(AbstractCommand::SUCCESS, $r['code']);
        self::assertCount(1, $http->calls, 'the second embed() is a cache hit');
        self::assertStringContainsString('8 dimensions, first values: 0.5, 0.5', $r['output']);
        self::assertStringNotContainsString(self::KEY, $r['output']);
    }

    public function test_probe_reports_a_provider_failure_without_a_stack_trace_or_the_key(): void
    {
        $http = (new RecordingHttpClient())->queue(new \AlfacodeTeam\PhpServicePlatform\Kernel\Ports\HttpClientResponse(401, '{"error":{"message":"bad ' . self::KEY . '","code":"invalid_api_key"}}'));
        $r    = $this->probe(EmbeddingsConfig::of(driver: 'openai', apiKey: self::KEY, dimensions: 8), $http, ['hello', '--live']);

        self::assertSame(AbstractCommand::FAILURE, $r['code']);
        self::assertStringContainsString('HTTP 401 (invalid_api_key)', $r['output']);
        self::assertStringNotContainsString(self::KEY, $r['output']);
    }

    private function unsetEnv(): void
    {
        unset($_ENV['EMBEDDINGS_DRIVER']);
    }
}
