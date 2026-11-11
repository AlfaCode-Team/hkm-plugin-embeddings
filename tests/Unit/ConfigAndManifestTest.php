<?php

declare(strict_types=1);

namespace Tests\Plugins\Embeddings\Unit;

use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\EmbeddingPort;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\Vector\EmbeddingException;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Plugins\Embeddings\Infrastructure\Config\EmbeddingsConfig;
use Plugins\Embeddings\Provider;

final class ConfigAndManifestTest extends TestCase
{
    /** @return array<string, mixed> */
    private static function manifest(): array
    {
        return json_decode((string) file_get_contents(\dirname(__DIR__, 2) . '/module.json'), true, flags: JSON_THROW_ON_ERROR);
    }

    /** @return array<string, array<string, mixed>> key => declaration */
    private static function declared(): array
    {
        $out = [];

        foreach (self::manifest()['config'] as $c) {
            $c = \is_array($c) ? $c : ['key' => $c];
            $out[$c['key']] = $c;
        }

        return $out;
    }

    // ── manifest ─────────────────────────────────────────────────────────────

    public function test_provider_mirrors_the_manifest(): void
    {
        $m = self::manifest();
        $p = new Provider();

        self::assertSame($m['solves'], $p->solves());
        self::assertSame($m['requires'], $p->requires());
        self::assertSame($m['exposes'], $p->exposes());
        self::assertSame(EmbeddingPort::class, $p->exposes()[0]);
        self::assertSame([], $m['routes']);
    }

    public function test_every_env_var_read_is_declared_and_every_declared_one_is_read(): void
    {
        $root = \dirname(__DIR__, 2);
        $read = [];

        foreach (new \RecursiveIteratorIterator(new \RecursiveDirectoryIterator($root . '/Infrastructure')) as $file) {
            if ($file->isFile() && $file->getExtension() === 'php') {
                $src = (string) file_get_contents($file->getPathname());
                preg_match_all("/env\\('([A-Z0-9_]+)'/", $src, $m);
                $read = [...$read, ...$m[1]];

                foreach (token_get_all($src) as $t) {
                    if (\is_array($t) && $t[0] === T_STRING && strtolower($t[1]) === 'getenv') {
                        self::fail($file->getFilename() . ' calls getenv(); use env()');
                    }
                }
            }
        }

        self::assertEqualsCanonicalizing(array_keys(self::declared()), array_values(array_unique($read)));
    }

    public function test_the_api_key_has_no_default_and_the_driver_has_no_default_and_is_required(): void
    {
        $d = self::declared();

        self::assertArrayNotHasKey('default', $d['EMBEDDINGS_API_KEY']);
        self::assertArrayNotHasKey('default', $d['EMBEDDINGS_DRIVER']);
        self::assertTrue($d['EMBEDDINGS_DRIVER']['required'], 'booting without choosing a driver must fail loudly');
        self::assertFalse($d['EMBEDDINGS_API_KEY']['required']);
    }

    public function test_defaults_are_declared_where_there_is_an_obvious_one(): void
    {
        $d = self::declared();

        self::assertSame(96, $d['EMBEDDINGS_BATCH_SIZE']['default']);
        self::assertSame(true, $d['EMBEDDINGS_CACHE']['default']);
        self::assertSame('https://api.openai.com/v1', $d['EMBEDDINGS_BASE_URL']['default']);
        // No default for model or dimensions: the right value depends on the driver.
        self::assertArrayNotHasKey('default', $d['EMBEDDINGS_MODEL']);
        self::assertArrayNotHasKey('default', $d['EMBEDDINGS_DIMENSIONS']);
    }

    // ── config ───────────────────────────────────────────────────────────────

    public function test_per_driver_defaults(): void
    {
        $local = EmbeddingsConfig::of(driver: 'local');
        self::assertSame(['local-hash-v1', 256, false], [$local->model, $local->dimensions, $local->dimensionsExplicit]);

        $openai = EmbeddingsConfig::of(driver: 'openai');
        self::assertSame(['text-embedding-3-small', 1536, false], [$openai->model, $openai->dimensions, $openai->dimensionsExplicit]);

        $explicit = EmbeddingsConfig::of(driver: 'openai', dimensions: 512);
        self::assertSame([512, true], [$explicit->dimensions, $explicit->dimensionsExplicit]);
    }

    /** @return iterable<string, array{string}> */
    public static function badDrivers(): iterable
    {
        yield 'empty'    => [''];
        yield 'unknown'  => ['cohere'];
        yield 'wrong case is normalised by fromEnv, not by of()' => ['OpenAI'];
    }

    #[DataProvider('badDrivers')]
    public function test_an_unknown_driver_is_refused_with_the_reason(string $driver): void
    {
        $this->expectException(EmbeddingException::class);
        $this->expectExceptionMessageMatches('/must be "local" or "openai".*no default on purpose/');

        EmbeddingsConfig::of(driver: $driver);
    }

    public function test_limits_are_clamped(): void
    {
        $c = EmbeddingsConfig::of(batchSize: 99999, timeout: 0, maxInputChars: -5, dimensions: 99999);

        self::assertSame(2048, $c->batchSize);
        self::assertSame(1, $c->timeout);
        self::assertSame(1, $c->maxInputChars);
        self::assertSame(2000, $c->dimensions, 'a vector store takes at most 2000');
        self::assertSame(1, EmbeddingsConfig::of(batchSize: 0)->batchSize);
    }

    public function test_ttl_zero_means_no_expiry(): void
    {
        self::assertNull(EmbeddingsConfig::of(cacheTtl: 0)->cacheTtl);
        self::assertSame(60, EmbeddingsConfig::of(cacheTtl: 60)->cacheTtl);
    }

    /** @return iterable<string, array{string|bool, bool}> */
    public static function cacheFlag(): iterable
    {
        yield 'false string' => ['false', false];
        yield '0'            => ['0', false];
        yield 'off'          => ['off', false];
        yield 'no'           => ['no', false];
        yield 'true string'  => ['true', true];
        yield '1'            => ['1', true];
        yield 'garbage keeps the default' => ['maybe', true];
    }

    #[DataProvider('cacheFlag')]
    public function test_the_cache_flag_is_parsed_not_cast(string|bool $raw, bool $expected): void
    {
        $saved = $_ENV['EMBEDDINGS_CACHE'] ?? null;
        $driver = $_ENV['EMBEDDINGS_DRIVER'] ?? null;

        try {
            $_ENV['EMBEDDINGS_CACHE']  = $raw;
            $_ENV['EMBEDDINGS_DRIVER'] = 'local';

            // (bool)"false" is true in PHP — this is the defect the parse exists to avoid.
            self::assertSame($expected, EmbeddingsConfig::fromEnv()->cache);
        } finally {
            $saved === null ? $this->unsetEnv('EMBEDDINGS_CACHE') : $_ENV['EMBEDDINGS_CACHE'] = $saved;
            $driver === null ? $this->unsetEnv('EMBEDDINGS_DRIVER') : $_ENV['EMBEDDINGS_DRIVER'] = $driver;
        }
    }

    public function test_from_env_normalises_the_driver_case_and_reads_the_key_lazily(): void
    {
        $keys = ['EMBEDDINGS_DRIVER', 'EMBEDDINGS_API_KEY'];
        $saved = array_map(static fn (string $k): mixed => $_ENV[$k] ?? null, $keys);

        try {
            $_ENV['EMBEDDINGS_DRIVER']  = ' OpenAI ';
            $_ENV['EMBEDDINGS_API_KEY'] = 'sk-first';

            $c = EmbeddingsConfig::fromEnv();
            self::assertSame('openai', $c->driver);
            self::assertSame('sk-first', $c->revealApiKey());

            // Read at USE time, not captured at construction: nothing about the key is held.
            $_ENV['EMBEDDINGS_API_KEY'] = 'sk-second';
            self::assertSame('sk-second', $c->revealApiKey());
        } finally {
            foreach ($keys as $i => $k) {
                $saved[$i] === null ? $this->unsetEnv($k) : $_ENV[$k] = $saved[$i];
            }
        }
    }

    public function test_the_key_is_in_no_dump_of_the_config(): void
    {
        $secret = 'sk-never-in-a-dump-123';
        $c      = EmbeddingsConfig::of(driver: 'openai', apiKey: $secret);

        self::assertStringNotContainsString($secret, print_r($c, true));
        self::assertStringNotContainsString($secret, var_export($c, true));
        self::assertStringNotContainsString($secret, (string) json_encode($c));
        self::assertStringNotContainsString($secret, json_encode((array) $c, JSON_PARTIAL_OUTPUT_ON_ERROR) ?: '');
        ob_start();
        var_dump($c);
        self::assertStringNotContainsString($secret, (string) ob_get_clean());

        try {
            serialize($c);
            self::fail('a closure-holding object must refuse serialisation');
        } catch (\Exception $e) {
            self::assertStringContainsString('Closure', $e->getMessage());
        }

        self::assertTrue($c->hasApiKey());
        self::assertFalse(EmbeddingsConfig::of(driver: 'openai')->hasApiKey());
    }

    private function unsetEnv(string $key): void
    {
        unset($_ENV[$key]);
    }
}
