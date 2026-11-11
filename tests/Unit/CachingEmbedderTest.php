<?php

declare(strict_types=1);

namespace Tests\Plugins\Embeddings\Unit;

use AlfacodeTeam\Ground\Fakes\FakeCache;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\CachePort;
use AlfacodeTeam\PhpServicePlatform\Kernel\Ports\Vector\EmbeddingException;
use PHPUnit\Framework\TestCase;
use Plugins\Embeddings\Application\CachingEmbedder;
use Tests\Plugins\Embeddings\Support\CountingEmbedder;

final class CachingEmbedderTest extends TestCase
{
    public function test_unchanged_text_is_never_embedded_twice(): void
    {
        $inner = new CountingEmbedder();
        $e     = new CachingEmbedder($inner, new FakeCache());

        $first  = $e->embed('the same words');
        $second = $e->embed('the same words');
        $third  = $e->embedMany(['the same words'])[0];

        self::assertSame(1, $inner->inputCount(), 'one provider input in total');
        self::assertCount(1, $inner->calls);
        self::assertSame($first->toArray(), $second->toArray(), 'a hit is BIT-identical: stored as packed float64');
        self::assertSame($first->toArray(), $third->toArray());
        self::assertSame(['hits' => 2, 'misses' => 1, 'providerCalls' => 1, 'providerInputs' => 1, 'cacheErrors' => 0], $e->stats());
    }

    public function test_a_changed_text_is_a_miss(): void
    {
        $inner = new CountingEmbedder();
        $e     = new CachingEmbedder($inner, new FakeCache());

        $a = $e->embed('hello');
        $b = $e->embed('hello!');

        self::assertSame(2, $inner->inputCount());
        self::assertNotSame($a->toArray(), $b->toArray());
    }

    public function test_a_batch_sends_only_the_misses_in_one_call_and_keeps_input_order(): void
    {
        $inner = new CountingEmbedder();
        $e     = new CachingEmbedder($inner, new FakeCache());

        $warm = $e->embedMany(['b', 'd']);           // warms b and d
        $inner->calls = [];

        $out = $e->embedMany(['a', 'b', 'c', 'd', 'e']);

        self::assertSame([['a', 'c', 'e']], $inner->calls, 'one call, misses only, first-seen order');
        self::assertCount(5, $out);
        self::assertSame($warm[0]->toArray(), $out[1]->toArray());
        self::assertSame($warm[1]->toArray(), $out[3]->toArray());

        // And every position holds ITS text's vector, not a neighbour's.
        $direct = new CountingEmbedder();
        foreach (['a', 'b', 'c', 'd', 'e'] as $i => $text) {
            self::assertSame($direct->embed($text)->toArray(), $out[$i]->toArray(), "position {$i} ({$text})");
        }
    }

    public function test_a_text_repeated_within_one_batch_is_embedded_once(): void
    {
        $inner = new CountingEmbedder();
        $e     = new CachingEmbedder($inner, new FakeCache());

        $out = $e->embedMany(['x', 'y', 'x', 'x']);

        self::assertSame([['x', 'y']], $inner->calls);
        self::assertSame($out[0]->toArray(), $out[2]->toArray());
        self::assertSame($out[0]->toArray(), $out[3]->toArray());
        // 4 inputs: two distinct misses, and the two repeats of "x" are hits on what the first paid for.
        self::assertSame(['hits' => 2, 'misses' => 2, 'providerCalls' => 1, 'providerInputs' => 2, 'cacheErrors' => 0], $e->stats());
    }

    /**
     * `isset($missing[$key])` in the lookup loop is redundant for CORRECTNESS — $missing is
     * keyed by hash, so a repeated text collapses on its own — and a revert check showed no
     * result-level test can see it. What it saves is a second cache round trip for the same
     * key, so that is what is pinned.
     */
    public function test_a_repeated_text_costs_one_cache_lookup_not_one_per_occurrence(): void
    {
        $cache = new class extends FakeCacheProxy {
            public int $gets = 0;

            public function get(string $key): mixed
            {
                $this->gets++;

                return parent::get($key);
            }
        };
        $e = new CachingEmbedder(new CountingEmbedder(), $cache);

        $e->embedMany(['x', 'x', 'x', 'y']);

        self::assertSame(2, $cache->gets, 'two distinct texts, two lookups');
    }

    public function test_an_all_hit_batch_makes_no_provider_call_and_an_empty_batch_none_either(): void
    {
        $inner = new CountingEmbedder();
        $e     = new CachingEmbedder($inner, new FakeCache());
        $e->embedMany(['a', 'b']);
        $inner->calls = [];

        $e->embedMany(['b', 'a']);
        self::assertSame([], $inner->calls);

        self::assertSame([], $e->embedMany([]));
        self::assertSame([], $inner->calls);
    }

    /**
     * The key, pinned. Expected value computed in Python:
     *   sha256(b"emb\0v1\0local-hash-v1\0256\0hello")
     */
    public function test_the_key_format_is_pinned_and_does_not_contain_the_text(): void
    {
        $e = new CachingEmbedder(new CountingEmbedder('local-hash-v1', 4), new FakeCache());

        // Dimension 4 here, so build the expectation for dims=4 by the same documented recipe in Python:
        //   sha256(b"emb\0v1\0local-hash-v1\0004\0hello") — see the two assertions below.
        self::assertSame('emb:' . hash('sha256', "emb\0v1\0local-hash-v1\0" . '4' . "\0hello"), $e->keyFor('hello'));

        $at256 = new CachingEmbedder(new class implements \AlfacodeTeam\PhpServicePlatform\Kernel\Ports\EmbeddingPort {
            public function embed(string $text): \AlfacodeTeam\PhpServicePlatform\Kernel\Ports\Vector\Vector { throw new \LogicException(); }
            public function embedMany(array $texts): array { throw new \LogicException(); }
            public function dimensions(): int { return 256; }
            public function model(): string { return 'local-hash-v1'; }
        }, new FakeCache());
        self::assertSame('emb:dae8a452f9611d58e2a1cf9706f86b9ffadf35ca994935b5f072c26b3af368dd', $at256->keyFor('hello'));

        self::assertStringNotContainsString('hello', $at256->keyFor('hello'));
    }

    public function test_the_key_changes_with_the_model_and_the_dimension_so_a_switch_starts_cold(): void
    {
        $cache = new FakeCache();

        $a = new CachingEmbedder(new CountingEmbedder('model-a', 4), $cache);
        $b = new CachingEmbedder(new CountingEmbedder('model-b', 4), $cache);
        $c = new CachingEmbedder(new CountingEmbedder('model-a', 8), $cache);

        self::assertNotSame($a->keyFor('t'), $b->keyFor('t'));
        self::assertNotSame($a->keyFor('t'), $c->keyFor('t'));

        // Same shared cache: model-b must not be served model-a's vector.
        $va = $a->embed('t');
        $vb = $b->embed('t');
        self::assertNotSame($va->toArray(), $vb->toArray());
        self::assertSame($vb->toArray(), (new CountingEmbedder('model-b', 4))->embed('t')->toArray());
    }

    public function test_the_ttl_is_applied_and_an_expired_entry_is_a_miss(): void
    {
        $cache = new FakeCache();
        $inner = new CountingEmbedder();
        $e     = new CachingEmbedder($inner, $cache, ttl: 100);

        $e->embed('t');
        $cache->clock()->travelSeconds(99);
        $e->embed('t');
        self::assertSame(1, $inner->inputCount(), 'still fresh at 99 s');

        $cache->clock()->travelSeconds(2);
        $e->embed('t');
        self::assertSame(2, $inner->inputCount(), 'expired after 101 s');
    }

    /** @return iterable<string, array{mixed}> */
    public static function corrupt(): iterable
    {
        yield 'not a string'        => [['x']];
        yield 'not base64'          => ['%%% not base64 %%%'];
        yield 'wrong length'        => [base64_encode(pack('d*', 1.0, 2.0))];
        yield 'right length, NaN'   => [base64_encode(pack('d*', NAN, 1.0, 1.0, 1.0))];
        yield 'empty string'        => [''];
    }

    #[\PHPUnit\Framework\Attributes\DataProvider('corrupt')]
    public function test_an_undecodable_or_wrong_length_entry_is_a_miss_not_a_wrong_vector(mixed $poison): void
    {
        $cache = new FakeCache();
        $inner = new CountingEmbedder();
        $e     = new CachingEmbedder($inner, $cache);

        $cache->set($e->keyFor('t'), $poison);

        $v = $e->embed('t');

        self::assertSame(1, $inner->inputCount(), 'recomputed');
        self::assertSame(4, $v->dimensions());
        self::assertSame($inner->embed('t')->toArray(), $v->toArray());
        self::assertIsString($cache->get($e->keyFor('t')), 'and the bad entry was overwritten with a good one');
    }

    public function test_a_cache_that_throws_is_a_miss_and_is_counted_never_an_outage(): void
    {
        $cache = new class extends FakeCacheProxy {
            public function get(string $key): mixed { throw new \RuntimeException('redis down'); }
            public function set(string $key, mixed $value, ?int $ttl = null): bool { throw new \RuntimeException('redis down'); }
        };
        $inner = new CountingEmbedder();
        $e     = new CachingEmbedder($inner, $cache);

        $a = $e->embed('t');
        $b = $e->embed('t');

        self::assertSame($a->toArray(), $b->toArray());
        self::assertSame(2, $inner->inputCount(), 'with no working cache every call goes through');
        self::assertSame(4, $e->stats()['cacheErrors'], '2 failed gets + 2 failed sets — and visible, so a dead cache is not mistaken for a cold one');
    }

    public function test_input_the_adapter_would_refuse_is_refused_even_when_everything_else_is_cached(): void
    {
        $inner = new CountingEmbedder();
        $e     = new CachingEmbedder($inner, new FakeCache(), maxInputChars: 10);

        foreach (['', '  ', str_repeat('x', 11)] as $bad) {
            try {
                $e->embed($bad);
                self::fail('refused');
            } catch (\InvalidArgumentException) {
            }
        }

        self::assertSame([], $inner->calls);
    }

    public function test_the_inner_returning_the_wrong_count_is_an_error_and_caches_nothing(): void
    {
        $broken = new class extends CountingEmbedder {
            public function embedMany(array $texts): array { return []; }
        };
        $cache = new FakeCache();
        $e     = new CachingEmbedder($broken, $cache);

        $this->expectException(EmbeddingException::class);
        $this->expectExceptionMessageMatches('/returned 0 vectors for 1 inputs/');

        try {
            $e->embed('t');
        } finally {
            self::assertSame([], $cache->keys());
        }
    }

    public function test_model_and_dimensions_are_the_inner_ones(): void
    {
        $e = new CachingEmbedder(new CountingEmbedder('m', 7), new FakeCache());

        self::assertSame('m', $e->model());
        self::assertSame(7, $e->dimensions());
    }
}

/** FakeCache is final; this satisfies CachePort by delegation so a test can make single methods throw. */
abstract class FakeCacheProxy implements CachePort
{
    private FakeCache $inner;

    public function __construct()
    {
        $this->inner = new FakeCache();
    }

    public function get(string $key): mixed { return $this->inner->get($key); }
    public function set(string $key, mixed $value, ?int $ttl = null): bool { return $this->inner->set($key, $value, $ttl); }
    public function delete(string $key): bool { return $this->inner->delete($key); }
    public function has(string $key): bool { return $this->inner->has($key); }
    public function remember(string $key, int $ttl, callable $callback): mixed { return $this->inner->remember($key, $ttl, $callback); }
    public function increment(string $key, int $by = 1): int { return $this->inner->increment($key, $by); }
    public function deletePattern(string $pattern): int { return $this->inner->deletePattern($pattern); }
    public function flush(): bool { return $this->inner->flush(); }
    public function lock(string $name, int $seconds = 0, ?string $owner = null): \AlfacodeTeam\PhpServicePlatform\Kernel\Ports\Lock { return $this->inner->lock($name, $seconds, $owner); }
    public function restoreLock(string $name, string $owner): \AlfacodeTeam\PhpServicePlatform\Kernel\Ports\Lock { return $this->inner->restoreLock($name, $owner); }
}
